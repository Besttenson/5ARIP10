# Scenario filtering README 

## File structure 
The following file structure shows the location of the relevant files. The repository includes much more files, but for the scenario extraction are not needed directly. 

```
Scenario filtering repository
├── waymax
│   ├── docs
│   │   ├── notebooks
│   │   │   ├── data_demo.ipynb
│   │   │   ├── datatypes_demo.ipynb
│   │   │   └── multi_actors_demo.ipynb
│   │   └── requirements.txt
│   ├── waymax
│   │   └── config.py
│   ├── file_renamer.py
│   ├── scenario_extraction.ipynb
│   ├── scenario_filtering.py (HPC use)
│   └── LICENSE
├── waymo_motion_scenario_mining
│   ├── utils
│   │   ├── runner.py
│   │   ├── runner_sc.py
│   │   ├── tags_generator.py
│   │   ├── runner_split.py (HPC use)
│   │   ├── parameters
│   │   │   ├── scenario_categories.py
│   │   │   └── tag_parameters.py
│   │   ├── logger
│   │   └── helpers
│   ├── results
│   │   ├── 2024_03_27-15_12 (example files)
│   │   │   ├── SC10 (example files)
│   │   │   │   └── Waymo_SC10.json (example files)
│   │   │   └── Waymo_tag.json (example files)
│   └── requirements.txt
└── README.md
```

## Running tagging and categorization code

1. Install the required scenario mining packages in a virtual environment from the [./waymo_motion_scenario_mining/requirements.txt](./waymo_motion_scenario_mining/requirements.txt) file.

2. To generate the tags run the following command

    ```shell
    python .\waymo_motion_scenario_mining\utils\runner.py --data_dir your_data_path
    ```

    This will automatically create a folder which is named by the time you run the code in [./waymo_motion_scenario_mining/results/](./waymo_motion_scenario_mining/results/) where the tags will be stored. An example of this folder and a few tags can be seen in [./waymo_motion_scenario_mining/results/2024-03-27-15_12/](./waymo_motion_scenario_mining/results/2024-03-27-15_12/).

    On the HPC the command can be changed to

    ```shell
    python .\waymo_motion_scenario_mining\utils\runner_split.py --data_dir your_data_path
    ```

    This will run the code over the amount of cores available, improving the tagging speed.
    

3. To categorize the scenarios by searching for a combination of tags, run the following command

   ```shell
   python .\waymo_motion_scenario_mining\utils\runner_sc.py --result_time your_result_time
   # MM-DD-HH_MM
   ```

   This will automatically create folders named by the names of your categories in [./waymo_motion_scenario_mining/results/](./waymo_motion_scenario_mining/results/) where the categorization files will be stored. An axample of this folder and a few tags can be seen in [./waymo_motion_scenario_mining/results/2024-03-27-15_12/SC10/](./waymo_motion_scenario_mining/results/2024-03-27-15_12/SC10/).

## Changing tagging and filtering parameters

To improve or finetune the results of the tagging and categorization it is possible to tune tag parameters and update scenario categories in the following places.

1. In [./waymo_motion_scenario_mining/utils/parameters/tag_parameters.py/](./waymo_motion_scenario_mining/utils/parameters/tag_parameters.py/) tag parameters can be changed to influence when specific tags are assigned to a scenario. For example, the minimum required yaw_rate for a turn can be changed. 

2. In [./waymo_motion_scenario_mining/utils/parameters/scenario_categories.py/](./waymo_motion_scenario_mining/utils/parameters/scenario_categories.py/) custom scenario categories can be made. This gives you the chance to design a specific combination of tags that is best for your usecase.

3. In [./waymo_motion_scenario_mining/utils/tags_generator.py/](./waymo_motion_scenario_mining/utils/tags_generator.py/) the tags are added to a scenario. Custom tags to better narrow down your scenario can be added here.

## Running the extra filtering code

1. The extra filtering code uses functions from the waymax repository, therefore the required Waymax packages need to be installed from [./waymax/docs/requirements.txt](./waymax/docs/requirements.txt)

2. When WOMD data is installed locally but does not include the whole dataset, it is required to run [./waymax/file_renamer.py](./waymax/file_renamer.py) to change the ending of each TFRecord file name. It is required to change the directory variable to the location of the WOMD files. 

3. The extra filtering code can be run and visualized in the [./waymax/scenario_extraction.ipynb](./waymax/scenario_extraction.ipynb) notebook.

    The extra filtering code can also be run from the [./waymax/scenario_filtering.py](./waymax/scenario_filtering.py) file for use on the HPC. 

    Note: it is required to add the location of the WOMD data and categorization files into both files befor use. Furthermore, at the end of the WOMD location it is required to include training_tfexample.tfrecord@## where the ## need to be replaced with the number of TFRecord files in the data folder.

## Chaning extra filtering parameters

The extra filtering parameters can be changed in the [./waymax/scenario_extraction.ipynb](./waymax/scenario_extraction.ipynb) or [./waymax/scenario_filtering.py](./waymax/scenario_filtering.py) files themselves.

## Configure access to Waymo Open Motion Dataset

Waymax is designed to work with the Waymo Open Motion dataset out of the box. Using gcloud it is possible to use the WOMD directly from the cloud in the filtering code.

A simple way to configure access is the following:

1.  Apply for [Waymo Open Dataset](https://waymo.com/open) access.

2.  Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)

3.  Run `gcloud auth login <your_email>` with the same email used for step 1.

4.  Run `gcloud auth application-default login`.

5.  Sometimes it might be needed to add a link directly to the location of your gcloud credentials before gaining access to the data. An example of how to do this is:

```shell
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\20192326\\AppData\\Roaming\\gcloud\\application_default_credentials.json
```

Please reference
[TF Datasets](https://www.tensorflow.org/datasets/gcs#authentication) for
alternative methods to authentication.

When using the WOMD dicrectly from the cloud it is required to use WOD_1_0_0_TRAINING, WOD_1_1_0_TRAINING or one of the other configs as the data_config variable instead of the DATA_LOCAL_01 config. These data configs can be found in [./waymax/waymax/config.py](./waymax/waymax/config.py).

## Getting familiar with waymax
A good way of getting familiar with waymax itself is by looking at the example notebooks located at [./waymax/docs/notebooks](./waymax/docs/notebooks/)
