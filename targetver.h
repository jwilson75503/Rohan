#pragma once

// The following macros define the minimum required platform.  The minimum required platform
// is the earliest version of Windows, Internet Explorer etc. that has the necessary features to run 
// your application.  The macros work by enabling all features available on platform versions up to and 
// including the version specified.

// Modify the following defines if you have to target a platform prior to the ones specified below.
// Refer to MSDN for the latest info on corresponding values for different platforms.
#ifndef _WIN32_WINNT            // Specifies that the minimum required platform is Windows Vista.
#define _WIN32_WINNT 0x0600     // Change this to the appropriate value to target other versions of Windows.
#endif

// http://stackoverflow.com/questions/7752404/shgetfolderpath-identifier-not-found
//#define WINVER 0x0601     // Kernel 6.1 == Windows 7/Server 2008 R2
//#define _WIN32_WINNT WINVER
//#define _WIN32_IE 0x0800  // 

//#define WINVER 0x0501
//#define _WIN32_WINNT WINVER
//#define _WIN32_IE 0x0600
//


#define NOMINMAX          // Don't define min and max (prefer C++ lib)
