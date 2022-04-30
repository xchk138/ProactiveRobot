// RealPlayAndPTZControl.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "RealPlayAndPTZControl.h"
#include "RealPlayAndPTZControlDlg.h"
#include "Logger.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CRealPlayAndPTZControlApp

BEGIN_MESSAGE_MAP(CRealPlayAndPTZControlApp, CWinApp)
	//{{AFX_MSG_MAP(CRealPlayAndPTZControlApp)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG
	ON_COMMAND(ID_HELP, CWinApp::OnHelp)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CRealPlayAndPTZControlApp construction

CString ConvertString(CString strText)
{
	char *val = new char[200];
	CString strIniPath,strRet;
	
	memset(val,0,200);
	GetPrivateProfileString("String",strText,"",
		val,200,"./langchn.ini");
	strRet = val;
	if(strRet.GetLength()==0)
	{
		//If there is no corresponding string in ini file then set it to be default value(English)
		strRet=strText;
	}
	delete val;
	return strRet;
}
//Set static text in dialogux box (English->current language)  
void g_SetWndStaticText(CWnd * pWnd)
{
	CString strCaption,strText;
	
	//Set main widnow title 
	pWnd->GetWindowText(strCaption);
	if(strCaption.GetLength()>0)
	{
		strText=ConvertString(strCaption);
		pWnd->SetWindowText(strText);
	}
	
	//Set small window title 
	CWnd * pChild=pWnd->GetWindow(GW_CHILD);
	CString strClassName;
	while(pChild)
	{
		//////////////////////////////////////////////////////////////////////////		
		//Added by Jackbin 2005-03-11
		strClassName = ((CRuntimeClass*)pChild->GetRuntimeClass())->m_lpszClassName;
		if(strClassName == "CEdit")
		{
			//The next small window 
			pChild=pChild->GetWindow(GW_HWNDNEXT);
			continue;
		}
		
		//////////////////////////////////////////////////////////////////////////	
		
		//Set current language text in small window
		pChild->GetWindowText(strCaption);
		strText=ConvertString(strCaption);
		pChild->SetWindowText(strText);
		
		//The next small window 
		pChild=pChild->GetWindow(GW_HWNDNEXT);
	}
}

CRealPlayAndPTZControlApp::CRealPlayAndPTZControlApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}

/////////////////////////////////////////////////////////////////////////////
// The one and only CRealPlayAndPTZControlApp object

CRealPlayAndPTZControlApp theApp;

/////////////////////////////////////////////////////////////////////////////
// CRealPlayAndPTZControlApp initialization

BOOL CRealPlayAndPTZControlApp::InitInstance()
{
	AfxEnableControlContainer();

	// Standard initialization
	// If you are not using these features and wish to reduce the size
	//  of your final executable, you should remove from the following
	//  the specific initialization routines you do not need.

#ifdef _AFXDLL
	Enable3dControls();			// Call this when using MFC in a shared DLL
#else
	Enable3dControlsStatic();	// Call this when linking to MFC statically
#endif

	// create a unique logger for the whole app
	Logger logger("video.log", MSG_DEBUG);
	ASSERT(logger.Available());
	logger.Info("App started");
	
	CRealPlayAndPTZControlDlg dlg(logger);
	logger.Info("UI started");

	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		// TODO: Place code here to handle when the dialog is
		//  dismissed with OK
	}
	else if (nResponse == IDCANCEL)
	{
		// TODO: Place code here to handle when the dialog is
		//  dismissed with Cancel
	}

	// Since the dialog has been closed, return FALSE so that we exit the
	//  application, rather than start the application's message pump.
	return FALSE;
}
