from detectron2.config import CfgNode as CN


def add_centernet_config(cfg):
    _C = cfg

    _C.MODEL.CENTERNET = CN()
    _C.MODEL.CENTERNET.NUM_CLASSES = 80
    _C.MODEL.CENTERNET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.CENTERNET.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.CENTERNET.PRIOR_PROB = 0.01
    _C.MODEL.CENTERNET.INFERENCE_TH = 0.05
    _C.MODEL.CENTERNET.CENTER_NMS = False
    _C.MODEL.CENTERNET.NMS_TH_TRAIN = 0.6
    _C.MODEL.CENTERNET.NMS_TH_TEST = 0.6
    _C.MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN = 1000
    _C.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN = 100
    _C.MODEL.CENTERNET.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 100
    _C.MODEL.CENTERNET.NORM = "GN"
    _C.MODEL.CENTERNET.USE_DEFORMABLE = False
    _C.MODEL.CENTERNET.NUM_CLS_CONVS = 4
    _C.MODEL.CENTERNET.NUM_BOX_CONVS = 4
    _C.MODEL.CENTERNET.NUM_SHARE_CONVS = 0
    _C.MODEL.CENTERNET.LOC_LOSS_TYPE = 'giou'
    _C.MODEL.CENTERNET.SIGMOID_CLAMP = 1e-4
    _C.MODEL.CENTERNET.HM_MIN_OVERLAP = 0.8
    _C.MODEL.CENTERNET.MIN_RADIUS = 4
    _C.MODEL.CENTERNET.SOI = [[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]]
    _C.MODEL.CENTERNET.POS_WEIGHT = 1.
    _C.MODEL.CENTERNET.NEG_WEIGHT = 1.
    _C.MODEL.CENTERNET.REG_WEIGHT = 2.
    _C.MODEL.CENTERNET.HM_FOCAL_BETA = 4
    _C.MODEL.CENTERNET.HM_FOCAL_ALPHA = 0.25
    _C.MODEL.CENTERNET.LOSS_GAMMA = 2.0
    _C.MODEL.CENTERNET.WITH_AGN_HM = False
    _C.MODEL.CENTERNET.ONLY_PROPOSAL = False
    _C.MODEL.CENTERNET.AS_PROPOSAL = False
    _C.MODEL.CENTERNET.IGNORE_HIGH_FP = -1.
    _C.MODEL.CENTERNET.MORE_POS = False
    _C.MODEL.CENTERNET.MORE_POS_THRESH = 0.2
    _C.MODEL.CENTERNET.MORE_POS_TOPK = 9
    _C.MODEL.CENTERNET.NOT_NORM_REG = True
    _C.MODEL.CENTERNET.NOT_NMS = False

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/lvis/lvis_v1_train_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT = 200
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 5
    _C.MODEL.BIFPN.NUM_BIFPN = 6
    _C.MODEL.BIFPN.NORM = 'GN'
    _C.MODEL.BIFPN.OUT_CHANNELS = 160
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.OUT_FEATURES = ['dla2']
    _C.MODEL.DLA.USE_DLA_UP = True
    _C.MODEL.DLA.NUM_LAYERS = 34
    _C.MODEL.DLA.MS_OUTPUT = False
    _C.MODEL.DLA.NORM = 'BN'
    _C.MODEL.DLA.DLAUP_IN_FEATURES = ['dla3', 'dla4', 'dla5']
    _C.MODEL.DLA.DLAUP_NODE = 'conv'

    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default'

    _C.DEBUG = False
    _C.SAVE_DEBUG = False
    _C.SAVE_PTH = False
    _C.VIS_THRESH = 0.3
    _C.DEBUG_SHOW_NAME = False