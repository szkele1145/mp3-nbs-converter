; Inno Setup 脚本：把 MP3NBS 转换器打成 Windows 安装程序
;
; 编译（需先装 Inno Setup 6）：
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
; 或：
;   iscc installer.iss
;
; 产物：dist\MP3NBS转换器_安装包.exe

#define AppName "MP3 音符盒转换器"
#define AppNameEn "MP3NBS Converter"
#define AppVersion "1.1.0"
#define AppPublisher "mp3-nbs-converter"
#define AppExeName "MP3NBS转换器.exe"

[Setup]
AppId={{6F1B2A54-3C7D-4E9A-8B15-2D4E6F8A1C33}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppNameEn} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppNameEn}
DefaultGroupName={#AppNameEn}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=MP3NBS转换器_安装包
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
SetupIconFile=assets\app.ico
LicenseFile=README.md

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Files]
; 目录版整包（启动 1-2 秒；单文件版每次启动要解包约 8 秒）
Source: "dist\MP3NBS转换器\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 说明书
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\使用说明"; Filename: "{app}\README.md"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "立即启动 {#AppName}"; Flags: nowait postinstall skipifsilent
