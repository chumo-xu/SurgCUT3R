from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes_Multi  # noqa
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .bedlam import BEDLAM_Multi
from .blendedmvs import BlendedMVS_Multi  # noqa
from .co3d import Co3d_Multi  # noqa
from .cop3d import Cop3D_Multi
from .dl3dv import DL3DV_Multi
from .dynamic_replica import DynamicReplica
from .eden import EDEN_Multi
from .hypersim import HyperSim_Multi
from .hoi4d import HOI4D_Multi
from .irs import IRS
from .mapfree import MapFree_Multi
from .megadepth import MegaDepth_Multi  # noqa
from .mp3d import MP3D_Multi
from .mvimgnet import MVImgNet_Multi
from .mvs_synth import MVS_Synth_Multi
from .omniobject3d import OmniObject3D_Multi
from .pointodyssey import PointOdyssey_Multi
from .realestate10k import RE10K_Multi
from .scannet import ScanNet_Multi
from .scannetpp import ScanNetpp_Multi  # noqa
from .smartportraits import SmartPortraits_Multi
from .spring import Spring
from .synscapes import SynScapes
from .tartanair import TartanAir_Multi
from .threedkb import ThreeDKenBurns
# from .uasol import UASOL_Multi  # 文件缺失，暂时注释
from .urbansyn import UrbanSyn
from .unreal4k import UnReal4K_Multi
from .vkitti2 import VirtualKITTI2_Multi  # noqa
from .waymo import Waymo_Multi  # noqa
from .wildrgbd import WildRGBD_Multi  # noqa
from .scared import Scared_Multi
from .scared_new import SCARED_Multi
from .scared_shuffled import SCARED_Shuffled
from .scared_new631 import SCARED_Multi631
from .scared_new2 import SCARED_Multinew2
from .scared_new4 import SCARED_Multinew4
from .scared_new7 import SCARED_Multinew7
from .scared_new10 import SCARED_Multinew10
from .scared_new11 import SCARED_Multinew11
from .scared_new12 import SCARED_Multinew12
from .scared_new14 import SCARED_Multinew14
from .scared_new15 import SCARED_Multinew15
from .scared_new17stage2 import SCARED_Multinew17stage2
from .scared_new18 import SCARED_Multinew18
from .scared_new19 import SCARED_Multinew19
from .scared_new100 import SCARED_Multinew100
from .scared_new_debug import SCARED_MultinewDebug
from .stereomis import STEREOMIS_Multi
from .c3vd import C3VD_Multi
from .stereomis import STEREOMIS_Multi
from .stereomis_stage2 import STEREOMIS_Multistage2
from accelerate import Accelerator



def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=accelerator.num_processes,
            fixed_length=fixed_length
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )

    return data_loader
