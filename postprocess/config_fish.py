from pathlib import Path


DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
SEED = 42


DATA_ROOT_FID = [Path("/work/nvme/bdne/fsemenzato/quijote_z0_fid")]
DATA_ROOT_DIFF = [Path("/work/nvme/bdne/fsemenzato/quijote_z0_diffs")]
DATA_ROOT_LHC = Path("/work/nvme/bdne/fsemenzato/quijote_z0")
DATA_ROOT_BSQ = Path("/work/nvme/bdne/fsemenzato/BSQ_128")
DATA_NAME = "df_m_128_PCS_z=0.npy"
PARAM_FILE = DATA_ROOT_LHC / "latin_hypercube_params.txt"


GRID_DIM = 128
BOX_SIZE = 1000.0  

PARAM_MEANS = [0.3175,
                0.049,
                0.6711,
                0.9624,
                0.834,
                ]


N_SIMS_FID = 10000      
N_SIMS_VARIED = 500     
N_SIMS_LHC = 2000
N_SIMS_BSQ = 2000
PARAMS_LHC = DATA_ROOT_LHC / "latin_hypercube_params.txt"
PARAMS_BSQ = DATA_ROOT_BSQ / "BSQ_params.txt"
PARAM_LABELS = ['Om', 's8', 'ns', 'h', 'Ob']
FOLDER_NAME_MAP = {"Om": "Om", "Ob": "Ob2", "h": "h", "ns": "ns", "s8": "s8"}

FIDUCIAL_PARAMS_INFO = {
    "Om": {"idx": 0, "step": 0.010},
    "Ob": {"idx": 1, "step": 0.002},
    "h":  {"idx": 2, "step": 0.020},
    "ns": {"idx": 3, "step": 0.020},
    "s8": {"idx": 4, "step": 0.015},
}

PARAM_INFO = {
    'Om': {'folder': 'Om', 'step': 0.01},
    'Ob': {'folder': 'Ob', 'step': 0.002},
    'h':  {'folder': 'h', 'step': 0.02},
    'ns': {'folder': 'ns', 'step': 0.02},
    's8': {'folder': 's8', 'step': 0.015},
}

DATA_ROOTS_LHC = [
    Path("/work/nvme/bdne/fsemenzato/quijote_z0"), 
]

N_SIMS_LHC = 2000