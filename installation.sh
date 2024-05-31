
# python -m venv pykan-env
# source pykan-env/bin/activate  # On Windows use `pykan-env\Scripts\activate`
pip install git+https://github.com/KindXiaoming/pykan.git
git clone https://github.com/KindXiaoming/pykan.git
# pip install -r pykan/requirements.txt

git clone https://github.com/hhyqhh/KAN-EA.git
cd KAN-EA
pip install -r requirements.txt
python3 setup.py develop

python3 run_kan_sps.py


git clone https://github.com/JianpanHuang/KAN.git
cd KAN-EA
pip install -r requirements.txt
python3 setup.py develop

python3 run_kan_sps.py
