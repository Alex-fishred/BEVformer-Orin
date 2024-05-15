import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import argparse
import torch
import mmcv
import copy
import numpy as np
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

import sys

sys.path.append(".")
from det2trt.utils.tensorrt import (
    get_logger,
    create_engine_context,
    allocate_buffers,
    do_inference,
)
from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset
import time
import os.path as osp

#我加的 為了可視化
from nuscenes.utils.data_classes import Box as NuScenesBox
import pyquaternion
#from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
import threading
from queue import Queue

DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
}

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", default="configs/bevformer/bevformer_small_trt.py",help="test config file path")
    parser.add_argument("trt_model", default="checkpoints/tensorrt/structure_prune_retrain_epoch_15.trt",help="checkpoint file")

    #測試用
    #parser.add_argument("--config", default="configs/bevformer/bevformer_small_trt.py",help="test config file path")
    #parser.add_argument("--trt_model", default="checkpoints/tensorrt/structure_prune_retrain_epoch_15.trt",help="checkpoint file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_tensorrt_plugin()

    trt_model = args.trt_model
    config_file = args.config
    TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)

    engine, context = create_engine_context(trt_model, TRT_LOGGER)

    stream = cuda.Stream()

    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib
        import sys

        sys.path.append(".")
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    output_shapes = config.output_shapes
    input_shapes = config.input_shapes
    default_shapes = config.default_shapes

    for key in default_shapes:
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = default_shapes[key]

    #測試val資料及
    #dataset = build_dataset(cfg=config.data.val)
    #測試Test資料集
    dataset = build_dataset(cfg=config.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )
    nusc = NuScenes(version=dataset.metadata['version'], dataroot='./data/nuscenes', verbose=True)
    pth_model = build_model(config.model, test_cfg=config.get("test_cfg"))

    ts = []
    bbox_results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    prev_bev = np.random.randn(config.bev_h_ * config.bev_w_, 1, config._dim_)
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }
    count=0
    for data in loader:
        img = data["img"][0].data[0].numpy()
        img_metas = data["img_metas"][0].data[0]

        use_prev_bev = np.array([1.0])
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = np.array([0.0])
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        can_bus = img_metas[0]["can_bus"]
        lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0)
        batch_size, cameras, _, img_h, img_w = img.shape

        output_shapes_ = {}
        for key in output_shapes.keys():
            shape = output_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            output_shapes_[key] = shape

        input_shapes_ = {}
        for key in input_shapes.keys():
            shape = input_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            input_shapes_[key] = shape

        inputs, outputs, bindings = allocate_buffers(
            engine, context, input_shapes=input_shapes_, output_shapes=output_shapes_
        )

        for inp in inputs:
            if inp.name == "image":
                inp.host = img.reshape(-1).astype(np.float32)
            elif inp.name == "prev_bev":
                inp.host = prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "use_prev_bev":
                inp.host = use_prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "can_bus":
                inp.host = can_bus.reshape(-1).astype(np.float32)
            elif inp.name == "lidar2img":
                inp.host = lidar2img.reshape(-1).astype(np.float32)
            else:
                raise RuntimeError(f"Cannot find input name {inp.name}.")

        trt_outputs, t = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        trt_outputs = {
            out.name: out.host.reshape(*output_shapes_[out.name]) for out in trt_outputs
        }

        prev_bev = trt_outputs.pop("bev_embed")
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        trt_outputs = {k: torch.from_numpy(v) for k, v in trt_outputs.items()}
        result=pth_model.post_process(**trt_outputs, img_metas=img_metas)
        #＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃可視化預測結果＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
        pred_result=format_results_FPS_show_use(result,dataset,count)
        keys=list(pred_result.keys())
        vis_img=Show_pred(keys[0],nusc2=nusc,pred_data=pred_result)
        cv2.namedWindow('BEV_Pred', cv2.WINDOW_NORMAL)  # 设置窗口属性为可调整大小
        #cv2.setWindowProperty('Generated Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # 将窗口设置为全屏模式
        cv2.imshow('BEV_Pred', vis_img)  # 显示当前帧
        #cv2.waitKey(1000)  # 等待1毫秒，确保窗口能够更新
        #cv2.destroyAllWindows()
        #cv2.imwrite('result/vis_test/'+str(count)+'.jpg', vis_img)
        # 检测键盘按键，如果按下 'q' 键则退出循环
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        ##############################儲存預測結果
        #正常版本
        #bbox_results.extend(pth_model.post_process(**trt_outputs, img_metas=img_metas))
        #我改的減少運算
        bbox_results.extend(result)
        
        ts.append(t)

        for _ in range(len(img)):
            prog_bar.update()
        count=count+1
        
    kwargs = {}
    kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
    print(kwargs['jsonfile_prefix'])
    #寫入模型結果到json檔案
    dataset.format_results(bbox_results, **kwargs)
    #要檢測test的時候就不需要開啟 測試mAP之類的指標
    #metric = dataset.evaluate(bbox_results)

    # summary
    ''' 要檢測test的時候不需要開啟指標評估
    print("*" * 50 + " SUMMARY " + "*" * 50)
    for key in metric.keys():
        if key == "pts_bbox_NuScenes/NDS":
            print(f"NDS: {round(metric[key], 3)}")
        elif key == "pts_bbox_NuScenes/mAP":
            print(f"mAP: {round(metric[key], 3)}")
    '''
    latency = round(sum(ts[1:-1]) / len(ts[1:-1]) * 1000, 2)
    print(f"Latency: {latency}ms")
    print(f"FPS: {1000 / latency}")


####################＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃##轉換結果(一幀一幀顯示結果專用)
def format_results_FPS_show_use(results,dataset2,count):
    """Format the results to json (standard format for COCO evaluation).

    Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.

    Returns:
        tuple: Returns (result_files, tmp_dir), where `result_files` is a \
            dict containing the json filepaths, `tmp_dir` is the temporal \
            directory created for saving json files when \
            `jsonfile_prefix` is not specified.
    """
    assert isinstance(results, list), 'results must be a list'
    #assert len(results) == len(self), (
    #    'The length of results is not equal to the dataset len: {} != {}'.
    #    format(len(results), len(self)))

    # currently the output prediction results could be in two formats
    # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
    # 2. list of dict('pts_bbox' or 'img_bbox':
    #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
    # this is a workaround to enable evaluation of both formats on nuScenes
    # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
    #if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
    #    result_files = self._format_bbox(results, jsonfile_prefix)
    if True:
        # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
        #result_files = dict()
        for name in results[0]:
            print(f'\nFormating bboxes of {name}')
            results_ = [out[name] for out in results]
            #tmp_file_ = osp.join(jsonfile_prefix, name)
            vis_result=_format_bbox(results_,dataset2,count)
    return vis_result


def _format_bbox(results, dataset3,count):
    """Convert the results to the standard format.

    Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str): The prefix of the output jsonfile.
            You can specify the output directory/filename by
            modifying the jsonfile_prefix. Default: None.

    Returns:
        str: Path of the output json file.
    """
    nusc_annos = {}
    mapped_class_names = dataset3.CLASSES

    print('Start to convert detection format...')
    #custom_sample_id=913
    for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
        annos = []
        boxes = output_to_nusc_box_test_use(det)
        #我改成用count 抓取 sample id 配合每帧輸出結果
        sample_token = dataset3.data_infos[count]['token']
        #原本作者寫的
        #sample_token = dataset3.data_infos[sample_id]['token']
        boxes = lidar_nusc_box_to_global_test_use(dataset3.data_infos[count], boxes,
                                            mapped_class_names,
                                            dataset3.eval_detection_configs,
                                            dataset3.eval_version)
        for i, box in enumerate(boxes):
            name = mapped_class_names[box.label]
            if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                ]:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = DefaultAttribute[name]
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = DefaultAttribute[name]

            nusc_anno = dict(
                sample_token=sample_token,
                translation=box.center.tolist(),
                size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(),
                detection_name=name,
                detection_score=box.score,
                attribute_name=attr)
            annos.append(nusc_anno)
        nusc_annos[sample_token] = annos
        #這是為了測試沙崙資料及
        #custom_sample_id=custom_sample_id+1
    '''
    nusc_submissions = {
        'meta': self.modality,
        'results': nusc_annos,
    }

    mmcv.mkdir_or_exist(jsonfile_prefix)
    res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
    print('Results writes to', res_path)
    mmcv.dump(nusc_submissions, res_path)
    '''
    #回傳 偵測結果
    return nusc_annos

def output_to_nusc_box_test_use(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global_test_use(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None,
                       nusc3=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc3.get('sample_data', sample_data_token)
    cs_record = nusc3.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc3.get('sensor', cs_record['sensor_token'])
    pose_record = nusc3.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc3.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic
def get_color(category_name: str,nusc4=None):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc4.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc4.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc4.colormap['movable_object.trafficcone']

    for key in nusc4.colormap.keys():
        if category_name in key:
            return nusc4.colormap[key]
    return (0, 0, 0)

def Show_pred(
        sample_toekn: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = None,
        pred_data=None,
        seg_list=None,
        nusc2=None
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    #assert sample_toekn + '.png' in seg_list, '分割图必须存在！'
    #lidar_img = lidiar_render(sample_toekn, pred_data, out_path=out_path)

    sample = nusc2.get('sample', sample_toekn)
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    result_data = []
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]

        sd_record = nusc2.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                         name=record['detection_name'], token='predicted') for record in
                     pred_data[sample_toekn] if record['detection_score'] > 0.2]

            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                                                                         box_vis_level=box_vis_level, pred_anns=boxes,nusc3=nusc2)
            _, boxes_gt, _ = nusc2.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
            data = cv2.imread(data_path)

            # Show boxes.
            if with_anns:
                for box in boxes_pred:
                    c = get_color(box.name,nusc4=nusc2)
                    box.render_cv2(data, view=camera_intrinsic, normalize=True, colors=(c, c, c))
                result_data.append(data)

        else:
            raise ValueError("Error: Unknown sensor modality!")

    # compose result data
    first_row = result_data[:3]
    second_row = result_data[3:]
    np.hstack(first_row)
    cam_img = np.vstack((np.hstack(first_row), np.hstack(second_row)))
    # compose seg map
    #假設要輸出seg map和鏡頭
    #seg_map = cv2.resize(lidar_img, (cam_img.shape[0], cam_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    #result_img = np.hstack((seg_map, cam_img))

    #假設不想要輸出seg map
    result_img=cam_img

    return result_img
    '''
    if out_path is not None:
        print(f"save_path: {out_path}.jpg")
        cv2.imwrite(out_path+'.jpg', result_img)
    '''


if __name__ == "__main__":
    main()
