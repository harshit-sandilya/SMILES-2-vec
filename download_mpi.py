import os
import urllib.request
import subprocess
import sys
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def elevate():
    """Re-run the script with admin privileges."""
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, " ".join(sys.argv), None, 1
    )
    sys.exit()

MSMPI_RUNTIME_URL = "https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisetup.exe"
MSMPI_SDK_URL     = "https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1/msmpisdk.msi"

RUNTIME_INSTALLER = "msmpisetup.exe"
SDK_INSTALLER     = "msmpisdk.msi"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def install_msmpi():
    download_file(MSMPI_RUNTIME_URL, RUNTIME_INSTALLER)
    download_file(MSMPI_SDK_URL,     SDK_INSTALLER)

    print("\nInstalling MS-MPI runtime...")
    subprocess.run([RUNTIME_INSTALLER, "-unattend"], check=True)
    print("MS-MPI runtime installed!")

    print("\nInstalling MS-MPI SDK...")
    subprocess.run(["msiexec", "/i", SDK_INSTALLER, "/quiet", "/norestart"], check=True)
    print("MS-MPI SDK installed!")

    print("\nReinstalling mpi4py...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "mpi4py", "-y"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install",   "mpi4py"],       check=True)
    print("mpi4py reinstalled!")

    os.remove(RUNTIME_INSTALLER)
    os.remove(SDK_INSTALLER)

    print("\n✔ MS-MPI installation complete!")
    print("Verify with: mpiexec --version")

if __name__ == "__main__":
    if not is_admin():
        print("Requesting administrator privileges...")
        elevate()
    else:
        install_msmpi()