cd /home/aistudio/data/

cp data122998/label_cls14_train.json .

rm -rf data12*

cd /home/aistudio/work/tools

python get_instance_for_bmn.py
python fix_bad_label.py



