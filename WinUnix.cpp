// OS-specific items handled here

/* Includes, cuda */
#include "stdafx.h"

/* Windows specific */
#define SECURITY_WIN32
#include <security.h>
#include <windows.h>
#include <secext.h>
#include <shfolder.h>
#include <shlobj.h>


int GetUserDocPath(char * sPath)
{mIDfunc
//WCHAR path[MAX_PATH];
	HRESULT hr = SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, sPath);
	return true;
}

int DirectoryEnsure(char * sPath)
{mIDfunc
/// checks that a directory exists and creates it if not
	if ((GetFileAttributes(sPath)) == INVALID_FILE_ATTRIBUTES){
		//cout << "Directory doesn't exist\n";
		CreateDirectory(sPath, 0);
		//cout << "Directory Created\n";
	}

	if ((GetFileAttributes(sPath)) == INVALID_FILE_ATTRIBUTES){
		return false;
	}
	else
		return true;
}

