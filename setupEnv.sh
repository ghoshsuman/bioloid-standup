#Questo file bash serve per configurare l'ambiente per il corretto funzionamento del progetto di Reiforcement Machine Learning di Vuotto
#Le fasi di configurazione sono 3: Generazione di un ambiente virtuale su cui installare python e V-REP, installazione di Carl e installazione di Stormpy

#Ho realizzato un file per ognuna di queste tre fasi di configurazione in modo che se una parte dovesse andare storta sarebbe più facile fare le opportune modifiche e ritentare la configurazione della specifica dipendenza

#Per evitare conflitti con eventuali altre versioni di python ecc... verrà creato un ambiente virtuale in cui tutte le dipendenze saranno installate

#ATTENZIONE! virtual env deve essere installato sul dispositivo

#crete the virtual environment
#bioloidVenv is the default name, you can change with the name you want to address to the environment
virtualenv --python=python3.5 venv

#con questo comando cambiamo il $PATH con il /bin di virtualenv
source venv/bin/activate
#per rimuovere questa modifica, basta eseguire:
# deactivate

#python install
sudo apt-get install -y build-essential checkinstall
sudo apt install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

#installazione di setupTools
python ez_setup.py

#scipy install
sudo apt-get install -y python3-scipy

#TODO install pytest

#pybrain install
easy_install matplotlib
git clone git://github.com/pybrain/pybrain.git pybrain
cd pybrain
python setup.py install
cd ..

#numpy install
sudo apt-get install python3-numpy

#V-REP

#download di V-REP, non è necessaria nessuna installazione una volta scompattato il file
wget http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_5_0_Linux.tar.gz
tar -xvzf V-REP_PRO_EDU_V3_5_0_Linux.tar.gz

#perchè funzioni bisogna copiare nella cartella del progetto (questa) tre file presenti nella cartella di V-REP
cp -f ./V-REP_PRO_EDU_V3_5_0_Linux/programming/remoteApiBindings/python/python/vrep.py ./

cp -f ./V-REP_PRO_EDU_V3_5_0_Linux/programming/remoteApiBindings/python/python/vrepConst.py ./

cp -f ./V-REP_PRO_EDU_V3_5_0_Linux/programming/remoteApiBindings/lib/lib/Linux/64Bit/remoteApi.so ./

mkdir -p libs
cd libs

#install boost
wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
tar -xvzf boost_1_66_0.tar.gz boost
rm boost_1_66_0.tar.gz
cd boost_1_66_0
./bootstrap.sh --prefix=/usr
sudo ./b2 install

./setupCarl.sh
./setupStormpy.sh
