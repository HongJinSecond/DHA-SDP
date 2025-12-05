This is a project for SDP, which use Lora to train specific adapter for LLM models to track SDP prediction.



# Environment

Python 3.10

Windows 11

# Structure

## ./baselines

This folder contains the two base line models we use: JITLine and CC2VEC.  Check the file `./baselines/results/result.csv` can get the experimental result of the two models.



## ./clusters

This folder contains our base cluster models. (K-Mean and Hierarchical).

`cluster_manager.py` and `feature_cluster.py` is the implementation of K-Mean. 

`HierarchicalCluster.py` is the implementation of Hierarchical Clustering.



## ./dataset

The dataset we use. 

files `changes_<>.pkl` are the semantic features (change code lines and commit message.)

files `features_<>.pkl` are the expert features (14 dimension).



## ./models

`CCT5.py` is the code for model cct5.

`ConcatModel.py` is the downstream model for SDP which will use the PLMs as encoder.

`FinalModel.py` is our final model for Style Aware.



## ./result

`./Kmean`, `developer_aware` and `project_final` are the model performance of the three style aware strategy. We finally save the average output for each PLM.

`RQ1_1` displays the result table for our RQ1.1.

`RQ1_2` displays the result table for our RQ1.2.

`RQ2` displays the result of our RQ2.

`RQ3` displays the result of our RQ3.

`Significance` is the result of our significance test.



## ./utils

This is the folder which contains the util script for our experiment.



# Run the code

## Base Model

***You can get our pre-trained models from xxxxxxx. Download it to the root directory, then you can start the testing phase directly without retraining.***

We suggest you use our pre-trained model, but if you want to train your own model, run the script as below:

```shell
python -W ignore base_train.py --do_train --base_train --epochs 10 --learning_rate 2e-5
```

The default PLM encoder is codeT5, you can modified the script of `base_train.py` in the part:

```python
if __name__ == "__main__":
    args = parse_jit_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # "concat" means we both use expert feature and semantic feature
    for base_model in ["concat"]:
        for pretrained in ["codet5"]:
            # You can add more pre-trained models like ["codebert","codet5", "graphcodebert", "unixcoder","plbart"]
            print(f"——————————————————run base model {base_model} on encoder {pretrained}————————————————————")
            args.base_model=base_model
            args.pretrained_model=pretrained
            main(args)
            torch.cuda.empty_cache()
```

The output model will be save in `./output/checkpoints/<model_name>`

If you want to test the performance of Base Model , use command:

```shell
python -W ignore base_train.py --do_test --base_train
```



**This file can not use the PLM CCT5 to train the Base Model**, our experiment directly use their pre-trained model, if you want to train our own CCT5 model, you can check their project  [CCT5](https://github.com/Ringbo/CCT5).



## Style Aware Model

Same as the process of training or testing, you can change the files to choose the PLMs you like to use. 

***The models we provide also includes trained LoRA models, so there is no need to retrain it here. You can directly run the test command script.***

**Data Distribution**

Train:

```shell
python -W ignore data_aware.py --do_train --lora_train --learning_rate 1e-4 --alpha 0.8
```

Test:

```shell
python -W ignore data_compare.py --do_test --lora_train
```

The file xxx_compare.py will run the test experiment and save the result in `./result/Kmean`



**Project Aware**

Train:

```shell
python -W ignore project_style_aware.py --do_train --lora_train --learning_rate 1e-4
```

Test:

```shell
python -W ignore project_style_compare.py --do_test --lora_train
```

The results can be found in `./result/project_final`



**Developer Aware**

Train:

```shell
python -W ignore developer_style_aware.py --do_train --lora_train --learning_rate 1e-4
```

Test:

```shell
python -W ignore developer_style_compare.py --do_test --lora_train
```

The results can be found in `./result/developer_aware`



**CCT5**

*Specifically, for the pre-trained model CCT5, since it directly uses the model provided by its authors, it is necessary to implement a separate experimental training code script as follows.*

Train:

```shell
python -W ignore cct5_run.py --do_train
```

Test

```shell
python -W ignore cct5_run.py --do_test
```



# 

