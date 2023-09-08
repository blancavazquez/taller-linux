echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Pipeline for classification model"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

python mnist.py --epochs 5 \
       --dropout 0.2 \
       --layers_dense 10 \
       --optimizer "adam" \
       --filename "experimento1"  
