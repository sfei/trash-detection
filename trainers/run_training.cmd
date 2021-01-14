
SET PIPELINE_CONFIG_PATH="pipeline.config"
SET MODEL_DIR="model"
SET NUM_TRAIN_STEPS=200000
SET SAMPLE_1_OF_N_EVAL_EXAMPLES=1
SET LOG_OUTPUT="output.log"

python model_main.py ^
--pipeline_config_path=%PIPELINE_CONFIG_PATH% ^
--model_dir=%MODEL_DIR% ^
--num_train_steps=%NUM_TRAIN_STEPS% ^
--sample_1_of_n_eval_examples=%SAMPLE_1_OF_N_EVAL_EXAMPLES% ^
--logtostderr 2> %LOG_OUTPUT%