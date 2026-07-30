Option Explicit

Dim shell, fso, baseDir, pythonw, launcher, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

If WScript.Arguments.Named.Exists("check") Then
    WScript.Quit 0
End If

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcher = fso.BuildPath(baseDir, "launch_app.pyw")
pythonw = fso.BuildPath(baseDir, ".venv\Scripts\pythonw.exe")

If Not fso.FileExists(pythonw) Then
    pythonw = "D:\software\miniconda\pythonw.exe"
End If

If Not fso.FileExists(pythonw) Then
    If shell.Run("cmd.exe /c where pythonw.exe >nul 2>nul", 0, True) = 0 Then
        pythonw = "pythonw.exe"
    Else
        MsgBox "Python was not found. Install Python 3.10 or newer, or create .venv.", 16, "YouTube Video Localizer"
        WScript.Quit 1
    End If
End If

shell.CurrentDirectory = baseDir
command = Chr(34) & pythonw & Chr(34) & " " & Chr(34) & launcher & Chr(34)
shell.Run command, 0, False
