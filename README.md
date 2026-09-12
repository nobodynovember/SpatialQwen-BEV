# SpatialQwen-BEV

SpatialQwen-BEV is a LLM-based agent designed for indoor zero-shot **Vision-and-Language Navigation (VLN)** task.

It extends [SpatialGPT](https://github.com/nobodynovember/SpatialGPT) by introducing a **unified topological–metric memory** on local Qwen LLM.
<p align="center">
  <img src="storybev.png" width="70%">
</p>

Key Features:
- **Spatio-Temporal Reasoning Chain**  
  A structured multi-step reasoning framework for navigation under partial observability.
- **BEV-based Navigation Memory (BNM):**  
  Metric memory records geometry, semantic, and trajectory information.
- **Directional Connected Landmark List (DCLL)**  
  Local topological memory encoding directional landmark relationships.
- **Spatial Knowledge Graph (SKG)**  
  Global topological memory capturing topology, explored regions, and object relations.



## 🎬 Demo of BEV Construction (Fast Playback)

<p align="center">
  <img src="Demo_BEV.gif" width="100%">
</p>
|------------------Orientation FPV-----------------|------Near-Field Semantic BEV----|-------Full-Scene Color BEV---------|



## 📦 Clone the Repository

```bash
git clone https://github.com/nobodynovember/SpatialQwen-BEV.git
cd SpatialQwen-BEV
```


## ⚙️ Installation

1. Matterport3D simulator docker installation instruction: [here](https://github.com/peteanderson80/Matterport3DSimulator)
2. Habitat-sim installation instruction: [here](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md)
3. Install dependencies:
```bash
conda create -n spatialgpt-bev python=3.10
conda activate spatialgpt-bev
pip install -r requirements.txt
```

## 📂 Data Preparation

1. To accelerate simulation, observation images should be pre-collected from the simulator. You can use your own saved images or use the [RGB_Observations.zip](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jadge_connect_hku_hk/Eq00RV04jXpNkwqowKh5mYABBTqBG1U2RXgQ7FvaGweJOQ?e=rL1d6p)  pre-collected in prior research work. Next, set DATA_ROOT in the spatialgpt-bev.sh file to point to the image directory.

2. To construct accurate BEVs, download the full MP3D dataset for Habitat as instructed [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset), and then link bevbuild/mp3d to this dataset path.

3. For the validation unseen set in experiment, follow [DUET](https://github.com/cshizhe/VLN-DUET/) to set the [annotations](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0) for testing on the val-unseen split. 

4. For the various-scenes set in experiment, once the above annotationss setting is complete, move the file SpatialGPTBEV_72_scenes_processed.json from current directory to the 'datasets/R2R/annotations/' directory.

5. For semantic grounding, download the LSeg checkpoint (demo_e200.ckpt) as instructed [here](https://github.com/isl-org/lang-seg), and place it under the bevbuilder/lseg/checkpoints directory .


## 🧠 Configure the Qwen LLM

SpatialQwen-BEV uses a locally deployed Qwen model through an OpenAI-compatible
API. Set `QWEN_BASE_URL` to the address of your Qwen service. The provided run
script uses `qwen3-vl-8b`.

No API key is required when the local Qwen endpoint has authentication disabled.
The client automatically uses `EMPTY` as the placeholder key expected by the
OpenAI Python SDK. If the endpoint address or authentication changes, configure
them with environment variables:

```bash
export QWEN_BASE_URL=http://localhost:<port>/v1
export QWEN_API_KEY=<key>  # optional; omit for an unauthenticated local server
```


## ▶️ Run SpatialQwen-BEV

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/SpatialQwen-BEV
bash scripts/spatialgpt-bev.sh
```

Note: If the MatterSim module is not found when running, rebuild MatterSim with Python 3.10 to ensure compatibility with the runtime environment. For example: cmake -DEGL_RENDERING=ON .. -DPYTHON_EXECUTABLE=/root/miniconda/envs/spatialgpt-bev/bin/python


## ⚙️ Key Parameters

```bash
--root_dir ${DATA_ROOT}
--img_root /path/to/images
--split SpatialGPTBEV_72_scenes_processed
--end 10 # the number of cases to be tested
--output_dir ${outdir}
--max_action_len 15
--save_pred
--stop_after 3
--llm qwen3-vl-8b
--response_format json
--max_tokens 4096
```



## 📬 Contact

If you have any questions, please contact:

zhiqiang.jiang@ucalgary.ca
