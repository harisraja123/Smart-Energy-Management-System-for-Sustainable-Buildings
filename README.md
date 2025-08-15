Install Python 3.9 if not installed already:

https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe

Set the path of Python in environment variables if its not already there.

https://www.mygreatlearning.com/blog/add-python-to-path/

Now, extract zip file to directory and open command prompt or powershell in same directory and run below commands.

1. Create New Virtual Enivornment

python -m venv autopilot

2. Activate Virtual Environment

autopilot\Scripts\activate

3. Install Other Packages and Libraries From requirements.txt

pip install -r requirements.txt

4. Uninstall Pytorch if it exists

pip uninstall torch torchvision torchaudio

5. Install PyTorch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

6. Dashboard Installation & Execution

pip install -r dashboard/requirements.txt

7. To run the project, pick any of below commands to do either training or testing of any one of the model.

# Training With Dashboard

python main.py --algorithm dqn --data_dir citylearn_dataset --with_dashboard --n_episodes 5000
python main.py --algorithm ppo --data_dir citylearn_dataset --with_dashboard --n_episodes 5000
python main.py --algorithm a3c --data_dir citylearn_dataset --with_dashboard --n_episodes 5000

# Training Without Running Dashboard

python main.py --algorithm dqn --data_dir citylearn_dataset --n_episodes 5000
python main.py --algorithm ppo --data_dir citylearn_dataset --n_episodes 5000
python main.py --algorithm a3c --data_dir citylearn_dataset --n_episodes 5000

# Testing of individual algorithm with dashboard

python main.py --mode test --algorithm dqn --test_episodes 10 --with_dashboard
python main.py --mode test --algorithm ppo --test_episodes 10 --with_dashboard
python main.py --mode test --algorithm a3c --test_episodes 10 --with_dashboard

# Testing of individual algorithm without running dashboard

python main.py --mode test --algorithm dqn --test_episodes 10
python main.py --mode test --algorithm ppo --test_episodes 10
python main.py --mode test --algorithm a3c --test_episodes 10

# Compare all algorithms with dashboard

python main.py --mode test --algorithm all --test_episodes 10 --with_dashboard

# Compare all algorithms without running dashboard

python main.py --mode test --algorithm all --test_episodes 10
