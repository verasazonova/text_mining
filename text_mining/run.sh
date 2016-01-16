#!/usr/bin/env bash
# This is a test running file for tweet_mining package
#
echo $#

BASE="/Users/verasazonova/Work"

MAIN_PY_BASE=$BASE"/PycharmProjects/text_mining/text_mining/experiments"
DATA_DIR=$BASE"/Data"
DATA_TYPE="imdb"

if [ "${DATA_TYPE}" == "tweets" ]; then
    DATA=$DATA_DIR"/tweet_sentiment/train.csv"
    DATA_TYPE="tweets"
    P_LAB="0.1"
    P_USED="0"
    CLASSIFICATION_TYPE="cv"
    #TEST_DATA=--test $DATA_DIR"/tweet_sentiment/test.csv"
elif [ "${DATA_TYPE}" == "imdb" ]; then
    DATA=$DATA_DIR"/imdb/train"
    UNLAB_DATA=$DATA_DIR"/imdb/train-unsup"
    TEST_DATA=--test $DATA_DIR"/imdb/test"
    P_LAB="1"
    P_USED="0"
    CLASSIFICATION_TYPE="test"
fi

PROGRAM_BASE=$BASE"/word2vec"
DATE=`date +%Y-%m-%d-%H-%M`

if [ "$#" -le 0 ]; then

    CUR_DIR=./$DATE
    mkdir $CUR_DIR

else
    CUR_DIR=./$1

fi

pushd $CUR_DIR

N_TRIAL=0

SIZE=100
WINDOW=10
MIN=1
SAMPLE=1e-3
NEGATIVE=0
ALPHA=0.025
ITER=20

DIFF1_MAX=10
DIFF0_MAX=0

CLF="lr"
MODEL_TYPE="w2v-sentence" # glove w2v-cbow gensim-skip-gram"
FORMAT='binary'

LOG="execution_log.txt"
touch $LOG

# train_data.txt test_data.txt unlabelled_data.txt
function run_sentences {

    cat $1 $2 $3 > alldata.txt


    L_TRAIN=`wc -l $1 | cut -d' ' -f1`
    L_TEST=`wc -l $2 | cut -d' ' -f1`
    echo $L_TRAIN, $L_TEST


    awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < alldata.txt > alldata-id.txt
    gshuf alldata-id.txt > alldata-id-shuf.txt

    time $PROGRAM_BASE/word2vec -train alldata-id-shuf.txt -output vectors.txt -cbow 0 -size $SIZE -window $WINDOW -negative 5 -hs 1 -sample $SAMPLE -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
    grep '_\*' vectors.txt | sed -e 's/_\*//' | sort -n > sentence_vectors.txt

    mv vectors.txt w2v_model

    head -n$L_TRAIN sentence_vectors.txt > x_train_vec_data.txt
    head -n$((L_TRAIN + L_TEST)) sentence_vectors.txt | tail -n$L_TEST > x_test_vec_data.txt


}

function run_glove {

    CORPUS=w2v_corpus.txt
    VOCAB_FILE=vocab.txt
    COOCCURRENCE_FILE=cooccurrence.bin
    COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
    BUILDDIR=$BASE/"GloVe-1.2/build"
    SAVE_FILE=w2v_model
    VERBOSE=2
    MEMORY=4.0
    MAX_ITER=$ITER
    BINARY=2
    NUM_THREADS=8

    X_MAX=$1
    ETA=$2
    ALPHA_PARAMETER=$3


    $BUILDDIR/vocab_count -min-count $MIN -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
    $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW < $CORPUS > $COOCCURRENCE_FILE
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    $BUILDDIR/glove -save-file TEMP -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -alpha $ALPHA_PARAMETER -eta $ETA -iter $MAX_ITER -vector-size $SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

    NLINES=`wc -l TEMP.txt | cut -d' '  -f1`
    echo $NLINES $SIZE > $SAVE_FILE
    cat TEMP.txt >> $SAVE_FILE
    rm TEMP.txt
}


# Read / split text data
# make for cv  --join_train_test
#
CMD="python $MAIN_PY_BASE/treat_data.py -f $DATA $UNLAB_DATA $TEST_DATA  --p_labeled $P_LAB --p_used $P_USED --ntrial $N_TRIAL --type $DATA_TYPE"
echo $CMD >> $LOG
$CMD
echo "Data read"

gshuf w2v_corpus.txt > TEMP
mv TEMP w2v_corpus.txt

# Build model:
if [ "${MODEL_TYPE}" == "w2v-skip-gram" ]; then
    ## with word2vec (mikolov)
    CMD="$PROGRAM_BASE/word2vec -train w2v_corpus.txt -output w2v_model -size $SIZE -window $WINDOW -min-count $MIN -sample $SAMPLE -negative $NEGATIVE -hs 1 -binary 1 -cbow 0 -iter $ITER -alpha $ALPHA"
    FORMAT='--binary'

elif [ "${MODEL_TYPE}" == "w2v-sentence" ]; then
    ## with word2vec sentence vectors (mikolov)

    run_sentences x_train.txt x_text.txt
    CMD=""
    FORMAT=''
    #FORMAT='--txt'

elif [ "${MODEL_TYPE}" == "w2v-cbow" ]; then
    ## with word2vec (mikolov)
    CMD="$PROGRAM_BASE/word2vec -train w2v_corpus.txt -output w2v_model -size $SIZE -window $WINDOW -min-count $MIN -sample $SAMPLE -negative $NEGATIVE -hs 1 -binary 1 -cbow 1 -iter $ITER -alpha $ALPHA"
    FORMAT='--binary'

elif [ "${MODEL_TYPE}" == "gensim-skip-gram" ]; then
    ## with gensim
    CMD="python $MAIN_PY_BASE/build_model.py --size $SIZE  --window $WINDOW --min $MIN --sample $SAMPLE --negative $NEGATIVE --iter $ITER --alpha $ALPHA"
    FORMAT='--binary'

elif [ "${MODEL_TYPE}" == "glove" ]; then

    run_glove 10 0.025 0.5   #$X_MAX $ETA $ALPHA_GLOVE
    CMD=""
    FORMAT=''

fi
echo $CMD >> $LOG
$CMD
echo "Model build"

#if [ "${MODEL_TYPE}" != "w2v-sentence" ]; then
    # Vectorize
CMD="python $MAIN_PY_BASE/vectorize_data.py --diff1_max $DIFF1_MAX  --diff0_max $DIFF0_MAX $FORMAT --sent_name sentence_vectors.txt"
echo $CMD >> $LOG
$CMD
echo "Data vectorized"
FORMAT=''
#fi

# Run classifier
OUTPUTNAME=${DATA##*/}
CMD="python $MAIN_PY_BASE/run_classification.py --dname ${OUTPUTNAME%%.*} $FORMAT --type $CLASSIFICATION_TYPE --clfbase $CLF --parameters $P_LAB $P_USED $N_TRIAL $SIZE $WINDOW $MIN $DIFF1_MAX $DIFF0_MAX $MODEL_TYPE $SAMPLE $ITER $ALPHA $NEGATIVE"
echo CMD >> $LOG
$CMD
echo "Classification completed"

#rm *.npy
#rm w2v_model

CMD="python $MAIN_PY_BASE/plot_experiment.py -f ${OUTPUTNAME%%.*}_lr"
echo CMD >> $LOG
$CMD
echo "Plotting completed"

popd

