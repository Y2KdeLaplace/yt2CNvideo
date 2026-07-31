Option Explicit

Dim shell, fso, baseDir, launcher, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

If WScript.Arguments.Named.Exists("check") Then
    WScript.Quit 0
End If

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcher = fso.BuildPath(baseDir, "launch_app.pyw")
If shell.Run("cmd.exe /c where uv.exe >nul 2>nul", 0, True) <> 0 Then
    MsgBox "uv was not found. Install uv first: https://docs.astral.sh/uv/", 16, "scip"
    WScript.Quit 1
End If

shell.CurrentDirectory = baseDir
command = "uv.exe run pythonw.exe " & Chr(34) & launcher & Chr(34)
shell.Run command, 0, False
