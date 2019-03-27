MODEL_DIR=./logs
TRAIN_DATA=./data/few_cols.csv
TRAINER_PACKAGE=trainer
TRAINER_MAIN=${TRAINER_PACKAGE}.classifier
BUCKET_NAME=ml_engbucket
JOB_NAME="classifier_$(shell date '+%Y%m%d_%H%M%S')"
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}
REGION=us-east1
RUNTIME_VERSION=1.8
TRAIN_DATA_CLOUD = gs://${BUCKET_NAME}/data/data.csv
args = `arg="$(filter-out $@,$(MAKECMDGOALS))"`
train_local:
	gcloud ml-engine local train \
	--module-name ${TRAINER_MAIN} \
	--package-path ./trainer \
	--job-dir ${MODEL_DIR} \
	-- \
	--train-file ${TRAIN_DATA}
train_job:
	gcloud ml-engine jobs submit training ${JOB_NAME} \
	--job-dir ${JOB_DIR} \
	--runtime-version ${RUNTIME_VERSION} \
	--module-name ${TRAINER_MAIN} \
	--package-path ./trainer \
	--region ${REGION} \
	-- \
	--train-file ${train_file}
make_predictions:
	gcloud ml-engine jobs submit prediction ${JOB_NAME} \
	--model classifier_model \
	--version v1 \
	--data-format TEXT \
	--region ${REGION} \
	--input-paths ${TRAIN_DATA_CLOUD} \
	--output-path ${JOB_DIR}/predictions
