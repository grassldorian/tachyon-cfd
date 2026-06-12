; Inno Setup script for Tachyon CFD
; Build the PyInstaller dist first:
;     pyinstaller packaging\RocketCFD.spec --noconfirm
; then compile this script (Inno Setup 6, https://jrsoftware.org/isinfo.php):
;     iscc packaging\tachyon_installer.iss
; Output: installer_out\TachyonCFD-Setup-<version>.exe

#define AppName "Tachyon CFD"
#define AppVersion "7.0"
#define AppExe "TachyonCFD.exe"

[Setup]
AppId={{7A3C9E1D-4B2F-4E8A-9C5D-TachyonCFD01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Tachyon CFD
DefaultDirName={autopf}\TachyonCFD
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=..\installer_out
OutputBaseFilename=TachyonCFD-Setup-{#AppVersion}
SetupIconFile=..\assets\tachyon.ico
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
WizardStyle=modern
; the CuPy bundle is large
DiskSpanning=no

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; \
    GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\TachyonCFD\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; \
    Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nTachyon CFD requires an NVIDIA GPU with a CUDA 12.x driver.
