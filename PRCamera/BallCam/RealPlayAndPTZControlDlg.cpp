// RealPlayAndPTZControlDlg.cpp : implementation file
//

#include "stdafx.h"
#include "RealPlayAndPTZControl.h"
#include "RealPlayAndPTZControlDlg.h"
#include "include/dhnetsdk.h"
#include "include/dhconfigsdk.h"
#include "PtzMenu.h"
#include "MultiPlay.h"
#include "MessageText.h"
#include <iostream>
#include <sstream>
#include <opencv.hpp>


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

//Forbid opening two programs at the same time 
#pragma data_seg("sharesec")
__declspec (allocate("sharesec")) HWND g_share_hWnd = NULL;
#pragma comment(linker,"/SECTION:sharesec,RWS")

#define WM_DEVICE_DISCONNECT	(WM_USER + 100)
#define WM_DEVICE_RECONNECT		(WM_USER + 101)

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About
CPlayAPI g_PlayAPI;

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	//{{AFX_DATA(CAboutDlg)
	enum { IDD = IDD_ABOUTBOX };
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAboutDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	//{{AFX_MSG(CAboutDlg)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
	//{{AFX_DATA_INIT(CAboutDlg)
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CRealPlayAndPTZControlDlg dialog

CRealPlayAndPTZControlDlg::CRealPlayAndPTZControlDlg(
	Logger & logger, 
	CWnd* pParent /*=NULL*/)
	: CDialog(CRealPlayAndPTZControlDlg::IDD, pParent), 
	m_logger(logger)
{
	//{{AFX_DATA_INIT(CRealPlayAndPTZControlDlg)
	m_DvrUserName = _T("admin");
	m_DvrPassword = _T("abcd-1234");
	m_DvrPort = 37777;
	m_presetData = 1;
	m_crviseGroup = 1;
	m_moveNo = 1;
	m_posX = 0;
	m_posY = 0;
	m_posZoom = 0;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_nChannelCount = 0;
	m_LoginID = 0;
	//Clear the 9-window real-time monitor handle 
	for(int i=0;i<9;i++)
	{
		m_DispHanle[i]=0;
		//Write 9-window information into CVideoNodeInfo array.
		m_videoNodeInfo[i].SetVideoInfo(i+1,"",0,-1,"","",DirectMode);
	}
	m_CurScreen =0;

	g_PlayAPI.LoadPlayDll();
}

void CRealPlayAndPTZControlDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CRealPlayAndPTZControlDlg)
	DDX_Control(pDX, IDC_COMBO_PLAYMODE, m_comboPlayMode);
	DDX_Control(pDX, IDC_COMBO_AUX_NO, m_auxNosel);
	DDX_Control(pDX, IDC_COMBO_PTZDATA, m_comboPTZData);
	DDX_Control(pDX, IDC_IRIS_OPEN, m_iris_open);
	DDX_Control(pDX, IDC_IRIS_CLOSE, m_iris_close);
	DDX_Control(pDX, IDC_FOCUS_FAR, m_focus_far);
	DDX_Control(pDX, IDC_FOCUS_NEAR, m_focus_near);
	DDX_Control(pDX, IDC_ZOOM_TELE, m_zoom_tele);
	DDX_Control(pDX, IDC_ZOOM_WIDE, m_zoom_wide);
	DDX_Control(pDX, IDC_PTZ_RIGHTDOWN, m_ptz_rightdown);
	DDX_Control(pDX, IDC_PTZ_RIGHTUP, m_ptz_rightup);
	DDX_Control(pDX, IDC_PTZ_LEFTDOWN, m_ptz_leftdown);
	DDX_Control(pDX, IDC_PTZ_LEFTUP, m_ptz_leftup);
	DDX_Control(pDX, IDC_PTZ_RIGHT, m_ptz_right);
	DDX_Control(pDX, IDC_PTZ_LEFT, m_ptz_left);
	DDX_Control(pDX, IDC_PTZ_DOWN, m_ptz_down);
	DDX_Control(pDX, IDC_PTZ_UP, m_ptz_up);
	DDX_Control(pDX, IDC_COMBO_Channel, m_comboChannel);
	DDX_Control(pDX, IDC_COMBO_DispNum, m_comboDispNum);
	DDX_Control(pDX, IDC_DvrIPAddress, m_DvrIPAddr);
	DDX_Text(pDX, IDC_EDIT_UserName, m_DvrUserName);
	DDX_Text(pDX, IDC_EDIT_Password, m_DvrPassword);
	DDX_Text(pDX, IDC_EDIT_Port, m_DvrPort);
	DDX_Text(pDX, IDC_PRESET_DATA, m_presetData);
//	DDV_MinMaxInt(pDX, m_presetData, 0, 100);
	DDX_Text(pDX, IDC_CRUISE_GROUP, m_crviseGroup);
	DDX_Text(pDX, IDC_MODE_NO, m_moveNo);
	DDX_Text(pDX, IDC_POS_X, m_posX);
	DDX_Text(pDX, IDC_POS_Y, m_posY);
	DDX_Text(pDX, IDC_POS_ZOOM, m_posZoom);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CRealPlayAndPTZControlDlg, CDialog)
	//{{AFX_MSG_MAP(CRealPlayAndPTZControlDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BT_Login, OnBTLogin)
	ON_BN_CLICKED(IDC_BUTTON_Play, OnBUTTONPlay)
	ON_BN_CLICKED(IDC_BT_Leave, OnBTLeave)
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_BUTTON_STOP, OnButtonStop)
	ON_BN_CLICKED(IDC_BTN_PTZEXCTRL, OnBtnPtzexctrl)
	ON_BN_CLICKED(IDC_PRESET_SET, OnPresetSet)
	ON_BN_CLICKED(IDC_PRESET_ADD, OnPresetAdd)
	ON_BN_CLICKED(IDC_PRESET_DELE, OnPresetDele)
	ON_BN_CLICKED(IDC_START_CRUISE, OnStartCruise)
	ON_BN_CLICKED(IDC_CRUISE_ADD_POINT, OnCruiseAddPoint)
	ON_BN_CLICKED(IDC_CRUISE_DEL_POINT, OnCruiseDelPoint)
	ON_BN_CLICKED(IDC_CRUISE_DEL_GROUP, OnCruiseDelGroup)
	ON_BN_CLICKED(IDC_MODE_SET_BEGIN, OnModeSetBegin)
	ON_BN_CLICKED(IDC_MODE_START, OnModeStart)
	ON_BN_CLICKED(IDC_MODE_SET_DELETE, OnModeSetDelete)
	ON_BN_CLICKED(IDC_LINE_SET_LEFT, OnLineSetLeft)
	ON_BN_CLICKED(IDC_LINE_SET_RIGHT, OnLineSetRight)
	ON_BN_CLICKED(IDC_LINE_START, OnLineStart)
	ON_BN_CLICKED(IDC_FAST_GO, OnFastGo)
	ON_BN_CLICKED(IDC_EXACT_GO, OnExactGo)
	ON_BN_CLICKED(IDC_RESET, OnResetZero)
	ON_BN_CLICKED(IDC_ROTATE_START, OnRotateStart)
	ON_BN_CLICKED(IDC_ROTATE_STOP, OnRotateStop)
	ON_BN_CLICKED(IDC_AUX_OPEN, OnAuxOpen)
	ON_BN_CLICKED(IDC_AUX_CLOSE, OnAuxClose)
	ON_BN_CLICKED(IDC_BTN_PTZMENU, OnBtnPtzmenu)
	ON_CBN_SELCHANGE(IDC_COMBO_DispNum, OnSelchangeCOMBODispNum)
	ON_CBN_CLOSEUP(IDC_COMBO_DispNum, OnCloseupCOMBODispNum)
	ON_MESSAGE(WM_DEVICE_DISCONNECT, OnDisConnect)
	ON_MESSAGE(WM_DEVICE_RECONNECT, OnReConnect)
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BUTTON_LIGHT_OPEN, &CRealPlayAndPTZControlDlg::OnBnClickedButtonLightOpen)
	ON_BN_CLICKED(IDC_BUTTON_LIGHT_CLOSE, &CRealPlayAndPTZControlDlg::OnBnClickedButtonLightClose)
	ON_BN_CLICKED(IDC_BUTTON_RAIN_BRUSH_OPEN, &CRealPlayAndPTZControlDlg::OnBnClickedButtonRainBrushOpen)
	ON_BN_CLICKED(IDC_BUTTON_RAIN_BRUSH_CLOSE, &CRealPlayAndPTZControlDlg::OnBnClickedButtonRainBrushClose)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CRealPlayAndPTZControlDlg message handlers

BOOL CRealPlayAndPTZControlDlg::OnInitDialog()
{
	//Forbid opening two programs at the same time 
	this->IsValid();
	
	CDialog::OnInitDialog();
	g_SetWndStaticText(this);
	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon
	
	// TODO: Add extra initialization here
	m_ptzScreen.Create(NULL,NULL,WS_CHILD|WS_VISIBLE,CRect(0,0,0,0),this,1981);
	UpdataScreenPos();
	m_ptzScreen.ShowWindow(SW_SHOW);
	m_ptzScreen.SetCallBack(MessageProcFunc,(LDWORD)this,
							GetParamsFunc,(LDWORD)this,
							SetParamsFunc,(LDWORD)this,
							RectEventFunc,(LDWORD)this);
	m_ptzScreen.SetShowPlayWin(SPLIT9,0);
	//Set initial IP address 
	m_DvrIPAddr.SetAddress(192,168,16,108);
	//Zoom dialogux box 
	CRect rectSeparator;
	GetWindowRect(&m_rectLarge);
	GetDlgItem(IDC_SEPERATOR)->GetWindowRect(&rectSeparator);
	m_rectSmall.left=m_rectLarge.left;
	m_rectSmall.top=m_rectLarge.top;
	m_rectSmall.right=m_rectLarge.right;
	m_rectSmall.bottom=rectSeparator.bottom;
	SetWindowPos(NULL,0,0,m_rectSmall.Width(),m_rectSmall.Height(),SWP_NOMOVE | SWP_NOZORDER);
	
	// setup UI font
	m_font.CreatePointFont(200, "Arial");
	GetDlgItem(IDC_DIST_INSIDE)->SetFont(&m_font);
	GetDlgItem(IDC_DIST_OUTSIDE)->SetFont(&m_font);

	//Command definition setup of PTZ control 
	InitPTZControl();
	m_logger.Info("Controller Initial Done");
	//Initialize channel and window dropdown menu 
	InitComboBox();
	//Initialize net SDK
	InitNetSDK();
	m_logger.Info("NetSDK Initial Done");

	// automatically login into the camera
	OnBTLogin();
	m_logger.Info("Login In Done");
	// setup the display channel
	m_comboChannel.SetCurSel(0);
	// setup the video stream mode as callback mode
	m_comboPlayMode.SetCurSel(0);
	// setup the screen number for the display region
	m_ptzScreen.SetShowPlayWin(0, 0);
	// click the play button
	OnBUTTONPlay();

	m_logger.Info("Playing video");

	m_postprocess_initialized = FALSE;
	m_canvas = NULL;
	m_is_moving_left = false;
	m_is_moving_right = false;
	m_is_alive = true;
	m_missing_counter = 0;

	// reset the camera state
	double tick_start, tick_stop, tick_freq;
	tick_freq = cv::getTickFrequency();
	tick_start = cv::getTickCount();
	tick_stop = tick_start;
	// begin zooming wider, so chessboard getting smaller
	PtzControl(DH_PTZ_ZOOM_DEC_CONTROL, FALSE);
	while (tick_stop - tick_start < tick_freq * 3) {
		// zoom camera for 3 seconds
		Sleep(100);
		tick_stop = cv::getTickCount();
	}
	PtzControl(DH_PTZ_ZOOM_DEC_CONTROL, TRUE);

	m_postprocessThread = std::thread(MainProcedure, this);

	return TRUE;  // return TRUE  unless you set the focus to a control
}

DWORD WINAPI MainProcedure(LPVOID _hDlg) {
	CRealPlayAndPTZControlDlg* dlg = (CRealPlayAndPTZControlDlg*)_hDlg;
	dlg->m_logger.Info("Entering MainProcedure");

	BITMAP _bm;

	while(dlg->m_is_alive){
		if (!dlg->m_postprocess_initialized) {
			dlg->m_postprocess_initialized = dlg->InitPostprocessHandler();
			dlg->m_logger.Info("Postprocess Thread Initialized");
		} else {
			// in case window size changed
			if (!dlg->m_canvas) return 0;
			dlg->m_bitmap = dlg->m_canvas->GetCurrentBitmap();
			if (!dlg->m_bitmap) return 0;
			dlg->m_bitmap->GetBitmap(&_bm);
			if (_bm.bmWidth != dlg->m_im_full.cols || _bm.bmHeight != dlg->m_im_full.rows) {
				dlg->m_logger.Warn("PANEL SIZE CHANGED!");
				dlg->m_postprocess_initialized = dlg->InitPostprocessHandler();
				dlg->m_logger.Info("Postprocess Thread Reinitialized!");
				continue;
			}
			// postprocess handler
			dlg->Postprocess();
			// visualize the results
			dlg->VisualizeResult();
			// send commands to follow targets
			dlg->FollowTarget();
		}
	}
	return 0;
}


BOOL CRealPlayAndPTZControlDlg::InitPostprocessHandler() {
	// setup postprocess handler
	m_canvas = GetDlgItem(IDC_SCREEN_WINDOW)->GetDC();
	m_logger.Debug("InitPostprocessHandler");
	if (m_canvas == NULL) {
		m_logger.Error("DC=NULL!");
		return FALSE;
	}
	
	m_bitmap = m_canvas->GetCurrentBitmap();
	
	if (m_bitmap == NULL) {
		m_logger.Error("BITMAP=NULL!");
		return FALSE;
	}

	BITMAP bm;
	m_bitmap->GetBitmap(&bm);

	int width, height, pitch, channel;
	width = bm.bmWidth;
	height = bm.bmHeight;
	pitch = bm.bmWidthBytes;

	if (width == 0 || height == 0 || pitch == 0) return FALSE;

	channel = pitch / width;

	{
		std::stringstream ss("");
		ss << "width=" << width << "; height=" << height
			<< "; pitch=" << pitch;
		if (pitch == width * 3) {
			ss << "; color=RGB";
		}
		else if (pitch == width * 4) {
			ss << "; color=RGBA";
		}
		else if (pitch == width) {
			ss << "; color=GRAY";
		}
		else return FALSE;
		m_logger.Debug(ss.str().c_str());
	}

	if (width * 4 != pitch) {
		m_logger.Error("channel != 4!");
		return FALSE;
	}

	m_size_ori.width = m_screenRect.Width();
	m_size_ori.height = m_screenRect.Height();

	m_rect_ori.x = m_screenRect.left + 10;
	m_rect_ori.y = m_screenRect.top + 38;
	m_rect_ori.width = m_screenRect.Width();
	m_rect_ori.height = m_screenRect.Height();

	m_im_full.create(cv::Size(width, height), CV_8UC4);
	m_im_ori.create(m_size_ori, CV_8UC4);
	m_im_gray.create(m_size_ori, CV_8UC1);

	{
		// print the camera rect
		std::stringstream ss("");
		ss.str("");
		ss << "rect: w=" << m_screenRect.Width() << "; h=" << m_screenRect.Height()
			<< "; x=" << m_screenRect.left << "; y=" << m_screenRect.top;
		m_logger.Debug(ss.str().c_str());
	}

	// image for visualizing chessboard inside 
	CWnd* pWndIn = GetDlgItem(IDC_FRAME_CB_INSIDE);

	if (!pWndIn) {
		m_logger.Error("pWndIn==NULL!");
		return FALSE;
	}
	
	CRect rect_win;
	GetWindowRect(&rect_win);

	CRect rect_vis_in;
	pWndIn->GetWindowRect(&rect_vis_in);
	
	m_rect_vis_in.x = rect_vis_in.left - rect_win.left;
	m_rect_vis_in.y = rect_vis_in.top - rect_win.top;
	m_rect_vis_in.width = rect_vis_in.Width();
	m_rect_vis_in.height = rect_vis_in.Height();

	{
		// print the IDC_FRAME_CB_IN rect
		std::stringstream ss("");
		ss.str("");
		ss << "cb_in: w=" << m_rect_vis_in.width << "; h=" << m_rect_vis_in.height
			<< "; x=" << m_rect_vis_in.x << "; y=" << m_rect_vis_in.y;
		m_logger.Debug(ss.str().c_str());
	}

	m_vis_in.create(cv::Size(m_rect_vis_in.size()), CV_8UC4);


	// image for visualizing chessboard outside 
	CWnd* pWndOut = GetDlgItem(IDC_FRAME_CB_OUTSIDE);

	if (!pWndOut) {
		m_logger.Error("pWndOut==NULL!");
		return FALSE;
	}

	CRect rect_vis_out;
	pWndOut->GetWindowRect(&rect_vis_out);
	m_rect_vis_out.x = rect_vis_out.left - rect_win.left;
	m_rect_vis_out.y = rect_vis_out.top - rect_win.top;
	m_rect_vis_out.width = rect_vis_out.Width();
	m_rect_vis_out.height = rect_vis_out.Height();

	{
		// print the IDC_FRAME_CB_OUT rect
		std::stringstream ss("");
		ss.str("");
		ss << "cb_out: w=" << m_rect_vis_out.width << "; h=" << m_rect_vis_out.height
			<< "; x=" << m_rect_vis_out.x << "; y=" << m_rect_vis_out.y;
		m_logger.Debug(ss.str().c_str());
	}

	m_vis_out.create(m_rect_vis_out.size(), CV_8UC4);
	
	return TRUE;
}


void CRealPlayAndPTZControlDlg::UpdataScreenPos()
{
	GetDlgItem(IDC_SCREEN_WINDOW)->GetClientRect(&m_clientRect);
	
	m_screenRect = m_clientRect;
	m_ptzScreen.MoveWindow(m_screenRect);
}

void CRealPlayAndPTZControlDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CRealPlayAndPTZControlDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CRealPlayAndPTZControlDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//Forbid opening two programs at the same time
void CRealPlayAndPTZControlDlg::IsValid()
{
	if(g_share_hWnd)
	{
		AfxMessageBox(ConvertString(_T("Only one program will be allowed to open!")));
		CWnd* pWnd = CWnd::FromHandle(g_share_hWnd);
		if(pWnd)
		{
			if (pWnd->IsIconic())
			{
				pWnd->ShowWindow(SW_RESTORE);
			}
			pWnd->SetForegroundWindow(); 
		}
		exit(0);
	}
	else
	{
		g_share_hWnd = m_hWnd;
	}
}

//Callback function when device disconnected
void CALLBACK DisConnectFunc(LLONG lLoginID, char *pchDVRIP, LONG nDVRPort, LDWORD dwUser)
{
	if(dwUser == 0)
	{
		return;
	}

	CRealPlayAndPTZControlDlg *pThis = (CRealPlayAndPTZControlDlg *)dwUser;
	HWND hWnd = pThis->GetSafeHwnd();
	if (NULL == hWnd)
	{
		return;
	}
	PostMessage(hWnd, WM_DEVICE_DISCONNECT, NULL, NULL);
}


void CALLBACK ReConnectFunc(LLONG lLoginID, char *pchDVRIP, LONG nDVRPort, LDWORD dwUser)
{
	if(dwUser == 0)
	{
		return;
	}

	CRealPlayAndPTZControlDlg *pThis = (CRealPlayAndPTZControlDlg *)dwUser;
	HWND hWnd = pThis->GetSafeHwnd();
	if (NULL == hWnd)
	{
		return;
	}
	PostMessage(hWnd, WM_DEVICE_RECONNECT, NULL, NULL);	
}

//Process when device disconnected 
LRESULT CRealPlayAndPTZControlDlg::OnDisConnect(WPARAM wParam, LPARAM lParam)
{
	//The codes need to be processed when device disconnected
	SetWindowText(ConvertString("RealPlayAndPTZControl disconnected"));
	return 0;
}

//Process when device reconnect
LRESULT CRealPlayAndPTZControlDlg::OnReConnect(WPARAM wParam, LPARAM lParam)
{
	//The codes need to be processed when device reconnect
	SetWindowText(ConvertString("RealPlayAndPTZControl"));
	return 0;
}


//Initialize net SDK
void CRealPlayAndPTZControlDlg::InitNetSDK()
{
	//Initialize net sdk, All callback begins here.
	BOOL bSuccess = CLIENT_Init(DisConnectFunc, (LDWORD)this);
	if (!bSuccess)
	{
		//Display function error occurrs reason.
		LastError();
	}	
	else
	{
		LOG_SET_PRINT_INFO  stLogPrintInfo = {sizeof(stLogPrintInfo)};
		CLIENT_LogOpen(&stLogPrintInfo);
		CLIENT_SetAutoReconnect(ReConnectFunc, (LDWORD)this);
	}
}

void CRealPlayAndPTZControlDlg::OnBTLogin() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);	//Get interface input 
	if(bValid)
	{
		char *pchDVRIP;
		CString strDvrIP = GetDvrIP();
		pchDVRIP = (LPSTR)(LPCSTR)strDvrIP;
		WORD wDVRPort=(WORD)m_DvrPort;
		char *pchUserName=(LPSTR)(LPCSTR)m_DvrUserName;
		char *pchPassword=(LPSTR)(LPCSTR)m_DvrPassword;

		NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY stInparam;
		memset(&stInparam, 0, sizeof(stInparam));
		stInparam.dwSize = sizeof(stInparam);
		strncpy(stInparam.szIP, pchDVRIP, sizeof(stInparam.szIP) - 1);
		strncpy(stInparam.szPassword, pchPassword, sizeof(stInparam.szPassword) - 1);
		strncpy(stInparam.szUserName, pchUserName, sizeof(stInparam.szUserName) - 1);
		stInparam.nPort = wDVRPort;
		stInparam.emSpecCap = EM_LOGIN_SPEC_CAP_TCP;

		NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY stOutparam;
		memset(&stOutparam, 0, sizeof(stOutparam));
		stOutparam.dwSize = sizeof(stOutparam);
		LLONG lRet = CLIENT_LoginWithHighLevelSecurity(&stInparam, &stOutparam);

		if(0 != lRet)
		{
			m_LoginID = lRet;
			GetDlgItem(IDC_BT_Login)->EnableWindow(FALSE);
			GetDlgItem(IDC_BT_Leave)->EnableWindow(TRUE);
			GetDlgItem(IDC_BUTTON_Play)->EnableWindow(TRUE);
			//Device channel and channel dropdown menu 
			int nRetLen = 0;
			NET_DEV_CHN_COUNT_INFO stuChn = { sizeof(NET_DEV_CHN_COUNT_INFO) };
			stuChn.stuVideoIn.dwSize = sizeof(stuChn.stuVideoIn);
			stuChn.stuVideoOut.dwSize = sizeof(stuChn.stuVideoOut);
			BOOL bRet = CLIENT_QueryDevState(lRet, DH_DEVSTATE_DEV_CHN_COUNT, (char*)&stuChn, stuChn.dwSize, &nRetLen);
			if (!bRet)
			{
				DWORD dwError = CLIENT_GetLastError() & 0x7fffffff;
			}			
	        m_nChannelCount = __max(stOutparam.stuDeviceInfo.nChanNum, stuChn.stuVideoIn.nMaxTotal);
			//m_nChannelCount = (int)deviceInfo.byChanNum;
			int nIndex = 0;
			m_comboChannel.ResetContent();
			for(int i=0;i<m_nChannelCount;i++)
			{
				CString str;
				str.Format("%d",i+1);
				nIndex = m_comboChannel.AddString(str);
				m_comboChannel.SetItemData(nIndex,i);
			}
			if(0 < m_comboChannel.GetCount())
			{
				nIndex = m_comboChannel.AddString(ConvertString("Multi_Preview"));
				m_comboChannel.SetItemData(nIndex,-1);
				m_comboChannel.SetCurSel(0);
			}
		}
		else
		{
			//Display log in failure reason 
			ShowLoginErrorReason(stOutparam.nError);
		}
	}
	SetWindowText(ConvertString("RealPlayAndPTZControl"));
}

//Play 
void CRealPlayAndPTZControlDlg::OnBUTTONPlay() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if (!bValid)
	{
		return;
	}

	int nIndex = m_comboDispNum.GetCurSel();
	if (CB_ERR == nIndex)
	{
		return;
	}

	int iDispNum =m_comboDispNum.GetItemData(nIndex);
	//Get video handle 
	HWND hWnd=GetDispHandle(iDispNum);
	if (NULL == hWnd)
	{
		return;
	}

	nIndex = m_comboChannel.GetCurSel();
	if (CB_ERR == nIndex)
	{
		return;
	}

	//Get channel number 
	int iChannel = m_comboChannel.GetItemData(nIndex);
	nIndex = m_comboPlayMode.GetCurSel();
	if (CB_ERR == nIndex)
	{
		return;
	}
	
	//Get play mode 
	RealPlayMode ePlayMode = (RealPlayMode)m_comboPlayMode.GetItemData(nIndex);
	if(-1 == iChannel && ePlayMode == DirectMode)
	{
		//Play directly in multiple-window preview 
		MultiPlayMode(iDispNum,hWnd);
	}
	if(-1 == iChannel && ePlayMode == ServerMode)
	{
		//Data callback play mode in multiple-window preview mode. 
		MultiPlayServerMode(iDispNum,hWnd);
	}
	if(-1 != iChannel && ePlayMode == DirectMode)
	{
		//Play directly 
		DirectPlayMode(iDispNum,iChannel,hWnd);
	}
	if(-1 != iChannel && ePlayMode == ServerMode)
	{
		//Play in data callback mode 
		ServerPlayMode(iDispNum,iChannel,hWnd);
	}
}

void CRealPlayAndPTZControlDlg::OnBTLeave() 
{
	// TODO: Add your control notification handler code here
	SetWindowText(ConvertString("RealPlayAndPTZControl"));
	if (0 == m_LoginID)
	{
		return;
	}

	//Clear 9-window real-time monitor handle 
	for(int i=0;i<9;i++)
	{
		CloseDispVideo(i + 1);
		m_DispHanle[i]=0;
		//Write 9-window message into CVideoNodeInfo array 
		m_videoNodeInfo[i].SetVideoInfo(i+1,"",0, -1,"","",DirectMode);
	}
	BOOL bSuccess = CLIENT_Logout(m_LoginID);
	if(bSuccess)
	{
		m_LoginID = 0;
		m_comboChannel.ResetContent();
		GetDlgItem(IDC_BT_Login)->EnableWindow(TRUE);
		GetDlgItem(IDC_BT_Leave)->EnableWindow(FALSE);
		GetDlgItem(IDC_BUTTON_Play)->EnableWindow(FALSE);
		Invalidate();
	}
	else
	{
		MessageBox(ConvertString("Fail to Logout!"), ConvertString("Prompt"));
	}	
}

//Get input IP
CString CRealPlayAndPTZControlDlg::GetDvrIP()
{
	CString strRet="";
	BYTE nField0,nField1,nField2,nField3;
	m_DvrIPAddr.GetAddress(nField0,nField1,nField2,nField3);
	strRet.Format("%d.%d.%d.%d",nField0,nField1,nField2,nField3);
	return strRet;
}

void CRealPlayAndPTZControlDlg::OnDestroy() 
{
	CDialog::OnDestroy();
	
	for(int i=0;i<9;i++)
	{
		CloseDispVideo(i + 1);
		m_DispHanle[i]=0;
	}

	// TODO: Add your message handler code here
	if(0 != m_LoginID)
	{
		CLIENT_Logout(m_LoginID);
	}
	//Clear SDK and then release occupied resources.
	CLIENT_Cleanup();
}

//Get video handle 
HWND CRealPlayAndPTZControlDlg::GetDispHandle(int nNum)
{
	HWND hWnd=0;
	hWnd = ((CWnd *)(m_ptzScreen.GetPage(m_CurScreen)))->m_hWnd;
	return hWnd;	
}

void CRealPlayAndPTZControlDlg::InitComboBox()
{
	//Video dropdown menu initialization 
	int nIndex;
	int i = 0;
	CString strDispNum[9]={"1","2","3","4","5","6","7","8","9"};
	m_comboDispNum.ResetContent();
	for(i=0;i<9;i++)
	{
		nIndex = m_comboDispNum.AddString(strDispNum[i]);
		m_comboDispNum.SetItemData(nIndex,i+1);
	}
	m_comboDispNum.SetCurSel(0);
	//Channel dropdown menu initialization 
	m_comboChannel.ResetContent();
	for(i=0;i<m_nChannelCount;i++)
	{
		CString str;
		str.Format("%d",i+1);
		nIndex = m_comboChannel.AddString(str);
		m_comboChannel.SetItemData(nIndex,i);
	}
	if(0 < m_comboChannel.GetCount())
	{
		nIndex = m_comboChannel.AddString(ConvertString("Multi_Preview"));
		m_comboChannel.SetItemData(nIndex,-1);
		m_comboChannel.SetCurSel(0);
	}
	//Control parameter dropdown menu initialization 
	CString strPTZData[8]={"1","2","3","4","5","6","7","8"};
	m_comboPTZData.ResetContent();
	for(i=0;i<8;i++)
	{
		nIndex = m_comboPTZData.AddString(strPTZData[i]);
		m_comboPTZData.SetItemData(nIndex,i+1);
	}
	m_comboPTZData.SetCurSel(5);
	//Auxiliary dropdown menu initialization
	CString strAuxNo[5]={"23","24","27","41","43"};
	m_auxNosel.ResetContent();
	for(i=0;i<5;i++)
	{
		m_auxNosel.AddString(strAuxNo[i]);
	}
	m_auxNosel.SetCurSel(0);

	//Play mode dropdown menu initialization 
	CString strPlayMode[3]={ConvertString("Direct-play"), ConvertString("Data-callback")};
	m_comboPlayMode.ResetContent();
	nIndex = m_comboPlayMode.AddString(strPlayMode[0]);
	m_comboPlayMode.SetItemData(nIndex,DirectMode);

	nIndex = m_comboPlayMode.AddString(strPlayMode[1]);
	m_comboPlayMode.SetItemData(nIndex,ServerMode);
	m_comboPlayMode.SetCurSel(0);
}

//Stop play
void CRealPlayAndPTZControlDlg::OnButtonStop() 
{
	// TODO: Add your control notification handler code here
	//Get video number 
	int nIndex = m_comboDispNum.GetCurSel();
	if(CB_ERR != nIndex)
	{
		int iDispNum = m_comboDispNum.GetItemData(nIndex);
		//Close current video 
		CloseDispVideo(iDispNum);
		//Refresh page 
		Invalidate();
		m_DispHanle[iDispNum-1] = 0;
	}
}

//Save video information 
void CRealPlayAndPTZControlDlg::SetPlayVideoInfo(int iDispNum,int iChannel,enum RealPlayMode ePlayMode)
{
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		CString strDvrIP = GetDvrIP();
		WORD wDVRPort=(WORD)m_DvrPort;
		m_videoNodeInfo[iDispNum-1].SetVideoInfo(iDispNum,strDvrIP,wDVRPort,iChannel,m_DvrUserName,m_DvrPassword,ePlayMode);
	}
}

//PTZ control 
BOOL CRealPlayAndPTZControlDlg::PtzControl(int type, BOOL stop)
{
	SetDlgItemText(IDC_PTZSTATUS,"");
	if (0 == m_LoginID)
	{
		return FALSE;
	}
	
	//Get channel number 
	UpdateData(TRUE);
	CString strDispNum;
	m_comboDispNum.GetWindowText(strDispNum);
	int iDispNum = atoi(strDispNum);
	int iChannel=m_videoNodeInfo[iDispNum-1].GetDvrChannel();
	if(-1 == iChannel)
	{
		return FALSE;
	}
	BYTE param1=0,param2=0;
	BYTE bPTZData=(unsigned char)GetDlgItemInt(IDC_COMBO_PTZDATA);
	switch(type) {
		case DH_PTZ_UP_CONTROL:
			//Up
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_DOWN_CONTROL:
			//Down
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_LEFT_CONTROL:
			//Left 
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_RIGHT_CONTROL:
			//Right 
			param1=0;
			param2=bPTZData;
			break;
		case DH_EXTPTZ_LEFTTOP:
			//Up left
			param1=bPTZData;
			param2=bPTZData;
			break;
		case DH_EXTPTZ_LEFTDOWN:
			//Up down 
			param1=bPTZData;
			param2=bPTZData;
			break;
		case DH_EXTPTZ_RIGHTTOP:
			//Up right 
			param1=bPTZData;
			param2=bPTZData;
			break;
		case DH_EXTPTZ_RIGHTDOWN:
			//Up down 
			param1=bPTZData;
			param2=bPTZData;
			break;
		case DH_PTZ_ZOOM_DEC_CONTROL:
			//Zoom out 
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_ZOOM_ADD_CONTROL:
			//Zoom in 
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_FOCUS_DEC_CONTROL:
			//Focus zoom in 
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_FOCUS_ADD_CONTROL:
			//Focus zoom out 
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_APERTURE_DEC_CONTROL:
			//Aperture zoom out 
			param1=0;
			param2=bPTZData;
			break;
		case DH_PTZ_APERTURE_ADD_CONTROL:
			//Aperture zoom in 
			param1=0;
			param2=bPTZData;
			break;
		default:
			return FALSE;
			//break;
	}
	BOOL bRet=CLIENT_DHPTZControl(m_LoginID,iChannel,type,param1,param2,0,stop);
	if(bRet)
	{
		SetDlgItemText(IDC_PTZSTATUS, ConvertString("Succeed"));
	}
	else
	{
		SetDlgItemText(IDC_PTZSTATUS, ConvertString("Fail"));
		m_logger.Warn("Fail to operate camera!");
	}
	if (type == DH_PTZ_LEFT_CONTROL || type == DH_PTZ_RIGHT_CONTROL) {
		char buf[32];
		sprintf_s(buf, "rotate L/R command done");
		m_logger.Debug(buf);
	}
	return bRet;
}

//Dialogue box zoom 
void CRealPlayAndPTZControlDlg::OnBtnPtzexctrl() 
{
	// TODO: Add your control notification handler code here
	CString str;
	if(GetDlgItemText(IDC_BTN_PTZEXCTRL,str),(str=="Advance>>" || str == ConvertString("Advance>>")))
	{
		SetDlgItemText(IDC_BTN_PTZEXCTRL, ConvertString("Close Advance<<"));
		SetWindowPos(NULL,0,0,m_rectLarge.Width(),m_rectLarge.Height(),SWP_NOMOVE | SWP_NOZORDER);
		GetDlgItem(IDC_SEPERATOR)->ShowWindow(SW_SHOW);
	}
	else
	{
		SetDlgItemText(IDC_BTN_PTZEXCTRL, ConvertString("Advance>>"));
		SetWindowPos(NULL,0,0,m_rectSmall.Width(),m_rectSmall.Height(),SWP_NOMOVE | SWP_NOZORDER);
		GetDlgItem(IDC_SEPERATOR)->ShowWindow(SW_HIDE);
	}
}

//PTZ control command definition setup 
void CRealPlayAndPTZControlDlg::InitPTZControl()
{
	m_ptz_up.SetButtonCommand(DH_PTZ_UP_CONTROL);
	m_ptz_down.SetButtonCommand(DH_PTZ_DOWN_CONTROL);
	m_ptz_left.SetButtonCommand(DH_PTZ_LEFT_CONTROL);
	m_ptz_right.SetButtonCommand(DH_PTZ_RIGHT_CONTROL);
	m_zoom_wide.SetButtonCommand(DH_PTZ_ZOOM_DEC_CONTROL);
	m_zoom_tele.SetButtonCommand(DH_PTZ_ZOOM_ADD_CONTROL);
	m_focus_near.SetButtonCommand(DH_PTZ_FOCUS_ADD_CONTROL);
	m_focus_far.SetButtonCommand(DH_PTZ_FOCUS_DEC_CONTROL);
	m_iris_open.SetButtonCommand(DH_PTZ_APERTURE_ADD_CONTROL);
	m_iris_close.SetButtonCommand(DH_PTZ_APERTURE_DEC_CONTROL);
	
	m_ptz_rightup.SetButtonCommand(DH_EXTPTZ_RIGHTTOP);
	m_ptz_rightdown.SetButtonCommand(DH_EXTPTZ_RIGHTDOWN);
	m_ptz_leftup.SetButtonCommand(DH_EXTPTZ_LEFTTOP);
	m_ptz_leftdown.SetButtonCommand(DH_EXTPTZ_LEFTDOWN);
}

//PTZ control extensive function 
void CRealPlayAndPTZControlDlg::PtzExtControl(DWORD dwCommand, DWORD dwParam)
{
	if (0 == m_LoginID)
	{
		return;
	}
	
	//Get channel number 
	CString strDispNum;
	CString strAuxNo;
	m_comboDispNum.GetWindowText(strDispNum);
	int iDispNum = atoi(strDispNum);
	int iChannel=m_videoNodeInfo[iDispNum-1].GetDvrChannel();
	if(-1 == iChannel)
	{
		return;
	}
	long param1=0,param2=0,param3 = 0;
	switch(dwCommand) {
		case DH_PTZ_POINT_MOVE_CONTROL:
			//Go to preset 
			param1=0;
			param2=(long)m_presetData;
			param3=0;
			break;
		case DH_PTZ_POINT_SET_CONTROL:
			//Add preset 
			param1=0;
			param2=(long)m_presetData;
			param3=0;
			break;
		case DH_PTZ_POINT_DEL_CONTROL:
			//Delete preset 
			param1=0;
			param2=(long)m_presetData;
			param3=0;
			break;
		case DH_PTZ_POINT_LOOP_CONTROL:
			//Begin tour/stop tour 
			if (0 == dwParam) 
			{
				param1=(long)m_crviseGroup;
				param2=0;
				param3=76;
			}
			else
			{
				param1=(long)m_crviseGroup;
				param2=0;
				param3=96;
			}
			break;
		case DH_EXTPTZ_ADDTOLOOP:
			//Add tour 
			param1=(long)m_crviseGroup;
			param2=(long)m_presetData;
			param3=0;
			break;
		case DH_EXTPTZ_DELFROMLOOP:
			//Delete tour 
			param1=(long)m_crviseGroup;
			param2=(long)m_presetData;
			param3=0;
			break;
		case DH_EXTPTZ_CLOSELOOP:
			//Delete tour group 
			param1=(long)m_crviseGroup;
			param2=0;//(long)m_presetData;
			param3=0;
			break;
		case DH_EXTPTZ_SETMODESTART:
			//begin record 
			param1=(long)m_moveNo;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_SETMODESTOP:
			//Stop record 
			param1=(long)m_moveNo;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_RUNMODE:
			//begin pattern 
			param1=(long)m_moveNo;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_STOPMODE:
			//Stop pattern 
			param1=(long)m_moveNo;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_DELETEMODE:
			//Delete pattern 
			param1=(long)m_moveNo;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_SETLEFTBORDER:
			//set left limit
			param1=0;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_SETRIGHTBORDER:
			//set right limit
			param1=0;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_STARTLINESCAN:
			//begin scan 
			param1=0;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_CLOSELINESCAN:
			//Stop scan 
			param1=0;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_FASTGOTO:
		case DH_EXTPTZ_EXACTGOTO:
			//3D intelligent position 
			param1=m_posX;
			param2=m_posY;
			param3=m_posZoom;
			break;
		case DH_EXTPTZ_RESETZERO:
			param1=0;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_STARTPANCRUISE:
			//Begin rotation
			param1=0;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_STOPPANCRUISE:
			//Stop rotation 
			param1=0;
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_AUXIOPEN:
			//Enable auxilian function 
			m_auxNosel.GetWindowText(strAuxNo);
			param1=atoi(strAuxNo);
			param2=0;
			param3=0;
			break;
		case DH_EXTPTZ_AUXICLOSE:
			//Stop auxilian function
			m_auxNosel.GetWindowText(strAuxNo);
			param1=atoi(strAuxNo);
			param2=0;
			param3=0;
			break;		
		default:
			break;
	}
	BOOL bRet=CLIENT_DHPTZControlEx(m_LoginID,iChannel,dwCommand,param1,param2,param3,FALSE);
	if(bRet)
	{
		SetDlgItemText(IDC_PTZSTATUS, ConvertString("Succeed"));
	}
	else
	{
		SetDlgItemText(IDC_PTZSTATUS, ConvertString("Fail"));
	}
}

//Go to preset 
void CRealPlayAndPTZControlDlg::OnPresetSet() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_PTZ_POINT_MOVE_CONTROL);
	}
}

//Add preset 
void CRealPlayAndPTZControlDlg::OnPresetAdd() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_PTZ_POINT_SET_CONTROL);
	}
}

//Delete preset 
void CRealPlayAndPTZControlDlg::OnPresetDele() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_PTZ_POINT_DEL_CONTROL);
	}
}

//Begin tour 
void CRealPlayAndPTZControlDlg::OnStartCruise() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		CString str;
		if(GetDlgItemText(IDC_START_CRUISE,str),(str == "Start Tour" || str == ConvertString("Start Tour")))
		{
			SetDlgItemText(IDC_START_CRUISE, ConvertString("Stop Tour"));
			PtzExtControl(DH_PTZ_POINT_LOOP_CONTROL, 0);
		}
		else
		{
			SetDlgItemText(IDC_START_CRUISE, ConvertString("Start Tour"));
			PtzExtControl(DH_PTZ_POINT_LOOP_CONTROL, 1);
		}
	}
}

//Add tour 
void CRealPlayAndPTZControlDlg::OnCruiseAddPoint() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_EXTPTZ_ADDTOLOOP);
	}
}

//Delete tour point
void CRealPlayAndPTZControlDlg::OnCruiseDelPoint() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_EXTPTZ_DELFROMLOOP);
	}
}

//Delete tour group 
void CRealPlayAndPTZControlDlg::OnCruiseDelGroup() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_EXTPTZ_CLOSELOOP);
	}
}

//Begin record 
void CRealPlayAndPTZControlDlg::OnModeSetBegin() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		CString str;
		if(GetDlgItemText(IDC_MODE_SET_BEGIN,str),(str == "Program Start" || str == ConvertString("Program Start")))
		{
			SetDlgItemText(IDC_MODE_SET_BEGIN, ConvertString("Program Stop"));
			PtzExtControl(DH_EXTPTZ_SETMODESTART);
		}
		else
		{
			SetDlgItemText(IDC_MODE_SET_BEGIN, ConvertString("Program Start"));
			PtzExtControl(DH_EXTPTZ_SETMODESTOP);
		}
	}
}

//Begin pattern 
void CRealPlayAndPTZControlDlg::OnModeStart() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		CString str;
		if(GetDlgItemText(IDC_MODE_START,str), (str =="Start Pattern" || str == ConvertString("Start Pattern")))
		{
			SetDlgItemText(IDC_MODE_START, ConvertString("Stop Pattern"));
			PtzExtControl(DH_EXTPTZ_RUNMODE);
		}
		else
		{
			SetDlgItemText(IDC_MODE_START, ConvertString("Start Pattern"));
			PtzExtControl(DH_EXTPTZ_STOPMODE);
		}
	}
}

//Delete pattern 
void CRealPlayAndPTZControlDlg::OnModeSetDelete() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_EXTPTZ_DELETEMODE);
	}
}

//Set left limit
void CRealPlayAndPTZControlDlg::OnLineSetLeft() 
{
	// TODO: Add your control notification handler code here
	PtzExtControl(DH_EXTPTZ_SETLEFTBORDER);
}

//Set right limit 
void CRealPlayAndPTZControlDlg::OnLineSetRight() 
{
	// TODO: Add your control notification handler code here
	PtzExtControl(DH_EXTPTZ_SETRIGHTBORDER);
}

//Begin scan 
void CRealPlayAndPTZControlDlg::OnLineStart() 
{
	// TODO: Add your control notification handler code here
	CString str;
	if(GetDlgItemText(IDC_LINE_START,str),(str == "Start" || str == ConvertString("Start")))
	{
		SetDlgItemText(IDC_LINE_START, ConvertString("Stop"));
		PtzExtControl(DH_EXTPTZ_STARTLINESCAN);	
	}
	else
	{
		SetDlgItemText(IDC_LINE_START, ConvertString("Start"));
		PtzExtControl(DH_EXTPTZ_CLOSELINESCAN);
	}
}

//3D intelligent position 
void CRealPlayAndPTZControlDlg::OnFastGo() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_EXTPTZ_FASTGOTO);
	}
}

void CRealPlayAndPTZControlDlg::OnExactGo() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_EXTPTZ_EXACTGOTO);
	}
}


void CRealPlayAndPTZControlDlg::OnResetZero() 
{
	// TODO: Add your control notification handler code here
	BOOL bValid = UpdateData(TRUE);
	if(bValid)
	{
		PtzExtControl(DH_EXTPTZ_RESETZERO);
	}
}


//Begin rotation 
void CRealPlayAndPTZControlDlg::OnRotateStart() 
{
	// TODO: Add your control notification handler code here
	PtzExtControl(DH_EXTPTZ_STARTPANCRUISE);
}

//Stop rotation 
void CRealPlayAndPTZControlDlg::OnRotateStop() 
{
	// TODO: Add your control notification handler code here
	PtzExtControl(DH_EXTPTZ_STOPPANCRUISE);
}

//Enable auxilian function 
void CRealPlayAndPTZControlDlg::OnAuxOpen() 
{
	// TODO: Add your control notification handler code here
	PtzExtControl(DH_EXTPTZ_AUXIOPEN);
}

//Close auxilian function
void CRealPlayAndPTZControlDlg::OnAuxClose() 
{
	// TODO: Add your control notification handler code here
	PtzExtControl(DH_EXTPTZ_AUXICLOSE);
}

void CRealPlayAndPTZControlDlg::OnBtnPtzmenu() 
{
	// TODO: Add your control notification handler code here
	if(0 == m_LoginID)
	{
		MessageBox(ConvertString("Please login first !"), ConvertString("Prompt"));
	}
	else
	{
		CPtzMenu dlg;
		CString strChannel;
		m_comboChannel.GetWindowText(strChannel);
		int iChannel=atoi(strChannel);
		// ��Ϊ�ı���Ϊ��1��ʼ������������ͨ������Ҫ����1
		if (iChannel > 0)
		{
			iChannel -= 1;
		}
		dlg.SetPtzParam(m_LoginID, iChannel);
		dlg.DoModal();
	}
}

//Play video directly 
void CRealPlayAndPTZControlDlg::DirectPlayMode(int iDispNum,int iChannel,HWND hWnd)
{
	//Close current video 
	CloseDispVideo(iDispNum);
	
	LLONG lRet = CLIENT_RealPlayEx(m_LoginID,iChannel,hWnd);
	if(0 != lRet)
	{
		m_DispHanle[iDispNum-1]=lRet;
		SetPlayVideoInfo(iDispNum,iChannel,DirectMode);
        CPlayWnd* pWnd = (CPlayWnd*)FromHandle(hWnd);
        pWnd->SetWndPlaying(true); 
	}
	else
	{
		MessageBox(ConvertString("Fail to play!"), ConvertString("Prompt"));
	}
}

//Play video in data callback mode 
void CRealPlayAndPTZControlDlg::ServerPlayMode(int iDispNum, int iChannel, HWND hWnd)
{	
	//Close current video 
	CloseDispVideo(iDispNum);
	
	//Enable stream
	BOOL bOpenRet = g_PlayAPI.PLAY_OpenStream(iDispNum,0,0,1024*512*6);
	if(bOpenRet)
	{
		//Begin play 
		BOOL bPlayRet = g_PlayAPI.PLAY_Play(iDispNum,hWnd);
		if(bPlayRet)
		{           
			//Real-time play 
			LLONG lRet = CLIENT_RealPlayEx(m_LoginID,iChannel,0);
			if(0 != lRet)
			{
				m_DispHanle[iDispNum-1]=lRet;
				SetPlayVideoInfo(iDispNum,iChannel,ServerMode);
				CLIENT_SetRealDataCallBackEx2(lRet, RealDataCallBackEx, (LDWORD)this, 0x0f);
                CPlayWnd* pWnd = (CPlayWnd*)FromHandle(hWnd);
                pWnd->SetWndPlaying(true); 
			}
			else
			{
				MessageBox(ConvertString("Fail to play!"), ConvertString("Prompt"));
				g_PlayAPI.PLAY_Stop(iDispNum);
				g_PlayAPI.PLAY_CloseStream(iDispNum);
			}
		}
		else
		{
			g_PlayAPI.PLAY_CloseStream(iDispNum);
		}
	}
	else
	{
		TRACE("PLAY_OpenStream failed, error: %d\n", g_PlayAPI.PLAY_GetLastError(iDispNum));
	}
}
								
void CALLBACK RealDataCallBackEx(LLONG lRealHandle, DWORD dwDataType, BYTE* pBuffer, DWORD dwBufSize, LLONG lParam, LDWORD dwUser)
{
	if (dwUser == 0) return;

	CRealPlayAndPTZControlDlg* dlg = (CRealPlayAndPTZControlDlg*)dwUser;
	dlg->ReceiveRealData(lRealHandle, dwDataType, pBuffer, dwBufSize, lParam);
}


BOOL CRealPlayAndPTZControlDlg::Postprocess() {
	// copy bitmap to opencv image
	size_t n_bytes = m_im_full.size().area() * m_im_full.channels();
	m_bitmap->GetBitmapBits(n_bytes, m_im_full.data);
	m_im_full(m_rect_ori).copyTo(m_im_ori);
	
	// check cost time
	double tick_freq = cv::getTickFrequency();
	double tick_beg = cv::getTickCount();
	double tick_end = tick_beg;

	// detection chessboard inside
	if (m_im_ori.empty()) return FALSE;
	if (!m_im_ori.size().area()) return FALSE;

	if (m_im_ori.channels() == 4)
		cv::cvtColor(m_im_ori, m_im_gray, cv::COLOR_BGRA2GRAY);
	else if (m_im_ori.channels() == 1)
		m_im_ori.copyTo(m_im_gray);
	else
	{
		m_logger.Error("unexpected image format!");
		return FALSE;
	}

	if (!DetectChessboard(CHESSBOARD_TYPE::CB_INSIDE)) {
		// chessboard detection failed
		m_cb_is_found_inside = false;
		m_cb_is_found_outside = false;
		
		//// detect chessboard inside the boat-house
		//if (!DetectChessboard(CHESSBOARD_TYPE::CB_OUTSIDE)) {
		//	m_cb_is_found_outside = false;
		//	m_logger.Info("cannot find chessboard!");
		//	return FALSE;
		//}
		//else m_cb_is_found_outside = true;
	}
	else {
		m_cb_is_found_inside = true;
		m_cb_is_found_outside = false;
	}

	tick_end = cv::getTickCount();
	int cost_milsec = int((tick_end - tick_beg) * 1000 / tick_freq);
	char buf_cost[32];
	sprintf_s(buf_cost, "cost: %d ms", cost_milsec);
	m_logger.Debug(buf_cost);

	return m_cb_is_found_inside || m_cb_is_found_outside;
}


void CRealPlayAndPTZControlDlg::VisualizeResult() {
	if (m_cb_is_found_outside) {
		char buf_print[8];
		sprintf_s(buf_print, "%2.03fm", m_dist_outside);
		SetDlgItemTextA(IDC_DIST_OUTSIDE, buf_print);
		SetDlgItemTextA(IDC_DIST_INSIDE, "--.--");

		//cv::resize(m_im_ori, m_vis_out, m_vis_out.size());
		//size_t n_bytes = m_im_full.size().area() * m_im_full.channels();
		//m_bitmap->GetBitmapBits(n_bytes, m_im_full.data);
		//m_vis_out.copyTo(m_im_full(m_rect_vis_out));
		//m_im_full(m_rect_vis_in) = cv::Scalar(0,0,0,0);
		//m_bitmap->SetBitmapBits(n_bytes, m_im_full.data);
	}
	else if(m_cb_is_found_inside) {
		char buf_print[8];
		sprintf_s(buf_print, "%2.03fm", m_dist_inside);
		SetDlgItemTextA(IDC_DIST_INSIDE, buf_print);
		SetDlgItemTextA(IDC_DIST_OUTSIDE, "--.--");

		//cv::resize(m_im_ori, m_vis_in, m_vis_in.size());
		//size_t n_bytes = m_im_full.size().area() * m_im_full.channels();
		//m_bitmap->GetBitmapBits(n_bytes, m_im_full.data);
		//m_vis_in.copyTo(m_im_full(m_rect_vis_in));
		//m_im_full(m_rect_vis_out) = cv::Scalar(0, 0, 0, 0);
		//m_bitmap->SetBitmapBits(n_bytes, m_im_full.data);
	}
	else {
		SetDlgItemTextA(IDC_DIST_INSIDE, "--.--");
		SetDlgItemTextA(IDC_DIST_OUTSIDE, "--.--");

		/*size_t n_bytes = m_im_full.size().area() * m_im_full.channels();
		m_bitmap->GetBitmapBits(n_bytes, m_im_full.data);
		m_im_full(m_rect_vis_in) = cv::Scalar(0, 0, 0, 0);
		m_im_full(m_rect_vis_out) = cv::Scalar(0, 0, 0, 0);
		m_bitmap->SetBitmapBits(n_bytes, m_im_full.data);*/
	}
}


//Process after receiving real-time data 
void CRealPlayAndPTZControlDlg::ReceiveRealData(LLONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, LLONG lParam)
{
	//Stream port number according to the real-time handle.
	LONG lRealPort=GetStreamPort(lRealHandle);
	if (0 == lRealPort)
	{
		return;
	}
	//Input the stream data getting from the card
	BOOL bInput=FALSE;
	switch(dwDataType) 
	{
		case 0:
			//Original data 
			bInput = g_PlayAPI.PLAY_InputData(lRealPort,pBuffer,dwBufSize);
			if (!bInput)
			{
				TRACE("input data error: %d\n", g_PlayAPI.PLAY_GetLastError(lRealPort));
			}
			break;
		case 1:
			//data with frame info 
	
			break;
		case 2:
			//yuv data 

			break;
		case 3:
			//pcm audio data 

			break;		
		default:
			break;
	}	
}

long CRealPlayAndPTZControlDlg::GetStreamPort(LLONG lRealHandle)
{
	long lPort=0;
	for(int i=0;i<9;i++)
	{
		if(lRealHandle == m_DispHanle[i])
		{
			lPort=i+1;
			break;
		}
	}
	return lPort;
}

//Close video directly 
void CRealPlayAndPTZControlDlg::StopPlayForDirectMode(int iDispNum)
{
	if(0 != m_DispHanle[iDispNum-1])
	{
		BOOL bRet = CLIENT_StopRealPlayEx(m_DispHanle[iDispNum-1]);
        HWND hWnd = GetDispHandle(iDispNum);
        CPlayWnd* pWnd = (CPlayWnd *)FromHandle(hWnd);
        pWnd->SetWndPlaying(false); 
	}
}

//Close video by data callback way 
void CRealPlayAndPTZControlDlg::StopPlayForServerMode(int iDispNum)
{
	//First close CLIENT_RealPlay
	if (0 == m_DispHanle[iDispNum-1])
	{
		return;
	}
	
	BOOL bRealPlay = CLIENT_StopRealPlayEx(m_DispHanle[iDispNum-1]);
	if(bRealPlay)
	{
        HWND hWnd = GetDispHandle(iDispNum);
        CPlayWnd* pWnd = (CPlayWnd *)FromHandle(hWnd);
        pWnd->SetWndPlaying(false); 

		//And then close PLAY_Play
		BOOL bPlay = g_PlayAPI.PLAY_Stop(iDispNum);
		if(bPlay)
		{
			//At last close PLAY_OpenStream
			BOOL bStream = g_PlayAPI.PLAY_CloseStream(iDispNum);
		}
	}
}

//Multiple-window preview 
void CRealPlayAndPTZControlDlg::MultiPlayMode(int iDispNum, HWND hWnd)
{
	//Close current video 
	CloseDispVideo(iDispNum);
	
	CMultiPlay dlg;
	dlg.SetMultiPlayDlgInfo(m_nChannelCount);
	int nResponse = dlg.DoModal();
	if (nResponse != IDOK)
	{
		return;
	}
	
	int nChannel = dlg.m_nChannel;
	int nMultiPlay = dlg.m_nMultiPlay;
	if (0 >= nMultiPlay || 0 == m_LoginID)
	{
		return;
	}
	

	if (!SetMultiSpliteMode(nChannel, nMultiPlay))
	{
		return;
	}

	LLONG lRet = 0;
	switch(nMultiPlay) 
	{
		case 1:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,hWnd,DH_RType_Multiplay_1);
			break;
		case 4:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,hWnd,DH_RType_Multiplay_4);
			break;
		case 8:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,hWnd,DH_RType_Multiplay_8);
			break;
		case 9:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,hWnd,DH_RType_Multiplay_9);
			break;
		case 16:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,hWnd,DH_RType_Multiplay_16);
			break;
		default:
			break;
	}
	if(0 != lRet)
	{
		m_DispHanle[iDispNum-1]=lRet;
		SetPlayVideoInfo(iDispNum,nChannel,MultiMode);
        CPlayWnd* pWnd = (CPlayWnd*)FromHandle(hWnd);
        pWnd->SetWndPlaying(true);  
	}
	else
	{
		MessageBox(ConvertString("Fail to play!"), ConvertString("Prompt"));
	} 
}

BOOL CRealPlayAndPTZControlDlg::SetMultiSpliteMode(int nChannel, int nMultiPlay)
{
	// ��ȡPrevViewMode
	CFG_EM_PREVIEW_MODE emPreviewMode = CFG_EM_PREVIEW_MODE_UNKNOWN;
	CFG_ENCODECAP stuEncodeCap = {0};
	
	char szBuf[1024 * 32] = {0};
	int nErr = 0;

	if (0 != m_LoginID)
	{
		BOOL bRet = CLIENT_QueryNewSystemInfo(m_LoginID, CFG_CMD_ENCODE_GETCAPS, -1, szBuf, sizeof(szBuf), &nErr, 5000);
		if (bRet)
		{
			int nRetLen = 0;
			bRet = CLIENT_ParseData(CFG_CMD_ENCODE_GETCAPS, szBuf, &stuEncodeCap, sizeof(stuEncodeCap), &nRetLen);
			if (bRet)
			{
				emPreviewMode = stuEncodeCap.emPreviewMode;
			}
		}
		
		if (CFG_EM_PREVIEW_MODE_SPLITSNAP == emPreviewMode)
		{
			int i = 0;
			DH_SPLIT_MODE emMode = (DH_SPLIT_MODE)-1;
			switch(nMultiPlay)
			{
			case 1:
				emMode = DH_SPLIT_1;
				break;
			case 4:
				emMode = DH_SPLIT_4;
				break;
			case 8:
				emMode = DH_SPLIT_8;
				break;
			case 9:
				emMode = DH_SPLIT_9;
				break;
			case 16:
				emMode = DH_SPLIT_16;
				break;
			default:
				break;
			}
			
			int nGroupId = nChannel/emMode;
			if (0 > nGroupId)
			{
				nGroupId = 0;
			}
			
			DH_SPLIT_MODE_INFO stuSplitModeInfo = {sizeof(stuSplitModeInfo)};
			stuSplitModeInfo.emSplitMode = emMode;
			stuSplitModeInfo.dwDisplayType = DH_SPLIT_DISPLAY_TYPE_GENERAL;
			stuSplitModeInfo.nGroupID = nGroupId;
			BOOL bRet = CLIENT_SetSplitMode(m_LoginID, 0, &stuSplitModeInfo, 5000);
			if (!bRet)
			{
				CString csErr;
				csErr.Format("CLIENT_SetSplitMode failed:%d", CLIENT_GetLastError() & 0x7ffffff);
				MessageBox(ConvertString(csErr), ConvertString("Prompt"));
				return FALSE;
			} 
		}
		
		return TRUE;
	}

	CString csErr;
	csErr.Format("Please login first !");
	MessageBox(ConvertString(csErr), ConvertString("Prompt"));
	return FALSE;
}

//Close video by multiple-window preview 
void CRealPlayAndPTZControlDlg::StopPlayForMultiMode(int iDispNum)
{
	if(0 != m_DispHanle[iDispNum-1])
	{
		BOOL bRet = CLIENT_StopRealPlayEx(m_DispHanle[iDispNum-1]);
        HWND hWnd = GetDispHandle(iDispNum);
        CPlayWnd* pWnd = (CPlayWnd *)FromHandle(hWnd);
        pWnd->SetWndPlaying(false); 
	}
}

//Display log in failure reason 
void CRealPlayAndPTZControlDlg::ShowLoginErrorReason(int nError)
{
	if(1 == nError)		MessageBox(ConvertString("Invalid password!"), ConvertString("Prompt"));
	else if(2 == nError)	MessageBox(ConvertString("Invalid account!"), ConvertString("Prompt"));
	else if(3 == nError)	MessageBox(ConvertString("Timeout!"), ConvertString("Prompt"));
	else if(4 == nError)	MessageBox(ConvertString("The user has logged in!"), ConvertString("Prompt"));
	else if(5 == nError)	MessageBox(ConvertString("The user has been locked!"), ConvertString("Prompt"));
	else if(6 == nError)	MessageBox(ConvertString("The user has listed into illegal!"), ConvertString("Prompt"));
	else if(7 == nError)	MessageBox(ConvertString("The system is busy!"), ConvertString("Prompt"));
	else if(9 == nError)	MessageBox(ConvertString("You Can't find the network server!"), ConvertString("Prompt"));
	else	MessageBox(ConvertString("Login failed!"), ConvertString("Prompt"));
}

//Display function execution error reason 
void CRealPlayAndPTZControlDlg::LastError()
{
	DWORD dwError = CLIENT_GetLastError();
	switch(dwError)
	{
	case NET_NOERROR:					GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NOERROR));
		break;
	case NET_ERROR:						GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_ERROR));
		break;
	case NET_SYSTEM_ERROR:				GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_SYSTEM_ERROR));
		break;
	case NET_NETWORK_ERROR:				GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NETWORK_ERROR));
		break;
	case NET_DEV_VER_NOMATCH:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_DEV_VER_NOMATCH));
		break;
	case NET_INVALID_HANDLE:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_INVALID_HANDLE));
		break;
	case NET_OPEN_CHANNEL_ERROR:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_OPEN_CHANNEL_ERROR));
		break;
	case NET_CLOSE_CHANNEL_ERROR:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_CLOSE_CHANNEL_ERROR));
		break;
	case NET_ILLEGAL_PARAM:				GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_ILLEGAL_PARAM));
		break;
	case NET_SDK_INIT_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_SDK_INIT_ERROR));
		break;
	case NET_SDK_UNINIT_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_SDK_UNINIT_ERROR));
		break;
	case NET_RENDER_OPEN_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_OPEN_ERROR));
		break;
	case NET_DEC_OPEN_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_DEC_OPEN_ERROR));
		break;
	case NET_DEC_CLOSE_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_DEC_CLOSE_ERROR));
		break;
	case NET_MULTIPLAY_NOCHANNEL:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_MULTIPLAY_NOCHANNEL));
		break;
	case NET_TALK_INIT_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_TALK_INIT_ERROR));
		break;
	case NET_TALK_NOT_INIT:				GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_TALK_NOT_INIT));
		break;	
	case NET_TALK_SENDDATA_ERROR:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_TALK_SENDDATA_ERROR));
		break;
	case NET_NO_TALK_CHANNEL:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NO_TALK_CHANNEL));
		break;
	case NET_NO_AUDIO:					GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NO_AUDIO));
		break;								
	case NET_REAL_ALREADY_SAVING:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_REAL_ALREADY_SAVING));
		break;
	case NET_NOT_SAVING:				GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NOT_SAVING));
		break;
	case NET_OPEN_FILE_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_OPEN_FILE_ERROR));
		break;
	case NET_PTZ_SET_TIMER_ERROR:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_PTZ_SET_TIMER_ERROR));
		break;
	case NET_RETURN_DATA_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RETURN_DATA_ERROR));
		break;
	case NET_INSUFFICIENT_BUFFER:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_INSUFFICIENT_BUFFER));
		break;
	case NET_NOT_SUPPORTED:				GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NOT_SUPPORTED));
		break;
	case NET_NO_RECORD_FOUND:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NO_RECORD_FOUND));
		break;	
	case NET_LOGIN_ERROR_PASSWORD:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_PASSWORD));
		break;
	case NET_LOGIN_ERROR_USER:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_USER));
		break;
	case NET_LOGIN_ERROR_TIMEOUT:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_TIMEOUT));
		break;
	case NET_LOGIN_ERROR_RELOGGIN:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_RELOGGIN));
		break;
	case NET_LOGIN_ERROR_LOCKED:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_LOCKED));
		break;
	case NET_LOGIN_ERROR_BLACKLIST:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_BLACKLIST));
		break;
	case NET_LOGIN_ERROR_BUSY:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_BUSY));
		break;
	case NET_LOGIN_ERROR_CONNECT:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_CONNECT));
		break;
	case NET_LOGIN_ERROR_NETWORK:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_LOGIN_ERROR_NETWORK));
		break;								
	case NET_RENDER_SOUND_ON_ERROR:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_SOUND_ON_ERROR));
		break;
	case NET_RENDER_SOUND_OFF_ERROR:	GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_SOUND_OFF_ERROR));
		break;
	case NET_RENDER_SET_VOLUME_ERROR:	GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_SET_VOLUME_ERROR));
		break;
	case NET_RENDER_ADJUST_ERROR:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_ADJUST_ERROR));
		break;
	case NET_RENDER_PAUSE_ERROR:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_PAUSE_ERROR));
		break;
	case NET_RENDER_SNAP_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_SNAP_ERROR));
		break;
	case NET_RENDER_STEP_ERROR:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_STEP_ERROR));
		break;
	case NET_RENDER_FRAMERATE_ERROR:	GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_RENDER_FRAMERATE_ERROR));
		break;
	case NET_CONFIG_DEVBUSY:			GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_CONFIG_DEVBUSY));
		break;
	case NET_CONFIG_DATAILLEGAL:		GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_CONFIG_DATAILLEGAL));
		break;							
	case NET_NO_INIT:					GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_NO_INIT));
		break;
	case NET_DOWNLOAD_END:				GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_DOWNLOAD_END));
		break;
	default:							GetDlgItem(IDC_LAST_ERROR)->SetWindowText(ConvertString(ERROR_NET_ERROR));						
	}
}

//Data callback mode in multiple-window preview 
void CRealPlayAndPTZControlDlg::MultiPlayServerMode(int iDispNum, HWND hWnd)
{
	//Close current video 
	CloseDispVideo(iDispNum);
	
	CMultiPlay dlg;
	dlg.SetMultiPlayDlgInfo(m_nChannelCount);
	int nResponse = dlg.DoModal();
	if (nResponse != IDOK)
	{
		return;
	}
	
	int nChannel = dlg.m_nChannel;
	int nMultiPlay = dlg.m_nMultiPlay;

	if (0 >= nMultiPlay || 0 == m_LoginID)
	{
		return;
	}
	
	//Enable stream

	BOOL bOpenRet = g_PlayAPI.PLAY_OpenStream(iDispNum,0,0,1024*512*6);
	if (!bOpenRet)
	{
		return;
	}

	//Begin play 
	BOOL bPlayRet = g_PlayAPI.PLAY_Play(iDispNum,hWnd);
	if (!bPlayRet)
	{
		g_PlayAPI.PLAY_CloseStream(iDispNum);
		return;
	}

	if (!SetMultiSpliteMode(nChannel, nMultiPlay))
	{
		g_PlayAPI.PLAY_Stop(iDispNum);
		g_PlayAPI.PLAY_CloseStream(iDispNum);
		return;
	}
	
	//Real-time play 
	LLONG lRet = 0;
	switch(nMultiPlay) 
	{
		case 1:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,0,DH_RType_Multiplay_1);
			break;
		case 4:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,0,DH_RType_Multiplay_4);
			break;
		case 8:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,0,DH_RType_Multiplay_8);
			break;
		case 9:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,0,DH_RType_Multiplay_9);
			break;
		case 16:
			lRet = CLIENT_RealPlayEx(m_LoginID,nChannel,0,DH_RType_Multiplay_16);
			break;
		default:
			break;
	}
	if(0 != lRet)
	{
		m_DispHanle[iDispNum-1]=lRet;
		SetPlayVideoInfo(iDispNum,nChannel,MultiServerMode);
		CLIENT_SetRealDataCallBackEx2(lRet, RealDataCallBackEx, (LDWORD)this, 0x0f);
        CPlayWnd* pWnd = (CPlayWnd*)FromHandle(hWnd);
        pWnd->SetWndPlaying(true); 
	}
	else
	{
		g_PlayAPI.PLAY_Stop(iDispNum);
		g_PlayAPI.PLAY_CloseStream(iDispNum);
		MessageBox(ConvertString("Fail to play!"), ConvertString("Prompt"));
	}
}

//Close video by data callback mode in multiple-window preview
void CRealPlayAndPTZControlDlg::StopPlayForMultiServer(int iDispNum)
{
	//First close CLIENT_RealPlay
	if(0 == m_DispHanle[iDispNum-1])
	{
		return;
	}
	
	BOOL bRealPlay = CLIENT_StopRealPlayEx(m_DispHanle[iDispNum-1]);
	if(bRealPlay)
	{
        HWND hWnd = GetDispHandle(iDispNum);
        CPlayWnd* pWnd = (CPlayWnd *)FromHandle(hWnd);
        pWnd->SetWndPlaying(false); 
		//Then close PLAY_Play
		BOOL bPlay = g_PlayAPI.PLAY_Stop(iDispNum);
		if(bPlay)
		{
			//At last close PLAY_OpenStream
			BOOL bStream = g_PlayAPI.PLAY_CloseStream(iDispNum);
		}
	}
}

//Close video 
void CRealPlayAndPTZControlDlg::CloseDispVideo(int iDispNum)
{
	//Close current video 
	enum RealPlayMode ePlayMode = m_videoNodeInfo[iDispNum-1].GetPlayMode();
	if(ePlayMode == DirectMode)
	{
		StopPlayForDirectMode(iDispNum);
	}
	else if(ePlayMode == ServerMode)
	{
		StopPlayForServerMode(iDispNum);
	}
	else if(ePlayMode == MultiMode)
	{
		StopPlayForMultiMode(iDispNum);
	}
	else if(ePlayMode == MultiServerMode)
	{
		StopPlayForMultiServer(iDispNum);
	}
}

/************************************************************************/
/* Callback function                                                      */
/************************************************************************/
void CALLBACK RectEventFunc(RECT WinRect,CPoint &pointStart,CPoint &pointEnd,LDWORD dwUser)
{
	CRealPlayAndPTZControlDlg *dlg =(CRealPlayAndPTZControlDlg *)dwUser;
	CPoint Origin;
	CPoint SendPoint;
	
	Origin.x = ( WinRect.left + WinRect.right ) / 2;
	Origin.y = ( WinRect.top + WinRect.bottom ) / 2;
	
	int dx = (pointStart.x + pointEnd.x)/2;
	int dy = (pointStart.y + pointEnd.y)/2;
	
	int width  = WinRect.right - WinRect.left;
	int height = WinRect.bottom - WinRect.top; 
	
	SendPoint.x = ( dx - Origin.x) * 8192 * 2 / width;
	SendPoint.y = ( dy - Origin.y) * 8192 * 2  / height ;
	
	int width2 = pointEnd.x - pointStart.x;
	int height2 = pointEnd.y - pointStart.y;
	int multiple = 0;

	if ( height2 !=0 && width2!=0)
	{
        if (pointEnd.y >= pointStart.y)
        {
            multiple = (width * height) / (width2 * height2);
        }
        else
        {
            multiple = -(width * height) / (width2 * height2);
        }
	}
	
	dlg->m_posX = SendPoint.x;
	dlg->m_posY = SendPoint.y;
	dlg->m_posZoom = multiple;
	dlg->UpdateData(FALSE);
	dlg->PtzExtControl(DH_EXTPTZ_FASTGOTO);
}
void CALLBACK  MessageProcFunc(int nWndID, UINT message, LDWORD dwUser)
{
	if(dwUser == 0)
	{
		return;
	}
	
	CRealPlayAndPTZControlDlg *dlg = (CRealPlayAndPTZControlDlg *)dwUser;
	dlg->MessageProc(nWndID, message);

}
BOOL CALLBACK  GetParamsFunc(int nWndID, int type, LDWORD dwUser)
{
	if(dwUser == 0)
	{
		return FALSE;
	}
	CRealPlayAndPTZControlDlg *dlg = (CRealPlayAndPTZControlDlg *)dwUser;
	return dlg->GetParams(nWndID, type);
}

BOOL CRealPlayAndPTZControlDlg::GetParams(int nWndID, int type)
{
	BOOL bRet = FALSE;
	
	if (type == 0)
	{
		bRet = GetExitDecode(nWndID);
	}
	else if (type == 1)
	{
		bRet = GetExitCycle(nWndID);
	}
	
	return bRet;
}

void CALLBACK SetParamsFunc(int nWndID, int type, LDWORD dwUser)
{
	if(dwUser == 0)
	{
		return;
	}
	
	CRealPlayAndPTZControlDlg *dlg = (CRealPlayAndPTZControlDlg *)dwUser;
	dlg->SetParams(nWndID, type);
}

void CRealPlayAndPTZControlDlg::SetParams(int nWndID, int type)
{
	if (type == 0)
	{
		SetExitDecode(nWndID);
	}
	else if (type == 1)
	{
		SetExitCycle(nWndID);
	}

}
void CRealPlayAndPTZControlDlg::MessageProc(int nWndID, UINT message)
{
	switch(message)
	{
	case WM_LBUTTONDOWN:
	case WM_RBUTTONDOWN:
		{
			m_CurScreen =nWndID;
			m_comboDispNum.SetCurSel(m_CurScreen);
			//SetCurWindId(nWndID);
			//UpdateCurScreenInfo();
		}
		break;
	default:
		break;
	}
	
}
BOOL CRealPlayAndPTZControlDlg::GetExitDecode(int nCurWndID)
{
	BOOL bRet = FALSE;
	if (nCurWndID<0 || nCurWndID>=16)
	{
		return bRet;
	}
	
	BOOL bIsTimeOut = m_cs.Lock();
	if(bIsTimeOut)
	{
		bRet = m_bWndExitDecode[nCurWndID];
		m_cs.Unlock();
	}
	
	return bRet;
}

BOOL CRealPlayAndPTZControlDlg::GetExitCycle(int nCurWndID)
{
	BOOL bRet = FALSE;
	if (nCurWndID<0 || nCurWndID>=16)
	{
		return bRet;
	}
	
	bRet = m_bWndExitCycle[nCurWndID];
	
	return bRet;
}

void CRealPlayAndPTZControlDlg::SetExitDecode(int nCurWndID)
{
	if (nCurWndID<0 || nCurWndID>=16)
	{
		return;
	}
	
	BOOL bIsTimeOut = m_cs.Lock();
	if(bIsTimeOut)
	{
		m_bWndExitDecode[nCurWndID] = !m_bWndExitDecode[nCurWndID];
		m_cs.Unlock();
	}
}

void CRealPlayAndPTZControlDlg::SetExitCycle(int nCurWndID)
{
	if (nCurWndID<0 || nCurWndID>=16)
	{
		return;
	}
	
	m_bWndExitCycle[nCurWndID] = !m_bWndExitCycle[nCurWndID];
}

void CRealPlayAndPTZControlDlg::OnSelchangeCOMBODispNum() 
{
	// TODO: Add your control notification handler code here
	m_CurScreen = m_comboDispNum.GetCurSel();
	m_ptzScreen.SetActiveWnd(m_CurScreen,TRUE);;
}

void CRealPlayAndPTZControlDlg::OnCloseupCOMBODispNum() 
{
	// TODO: Add your control notification handler code here
	
}

void CRealPlayAndPTZControlDlg::OnBnClickedButtonLightOpen()
{
	// TODO: Add your control notification handler code here
	AuxFunctionOperate(DH_EXTPTZ_LIGHTCONTROL, false);
}

void CRealPlayAndPTZControlDlg::OnBnClickedButtonLightClose()
{
	// TODO: Add your control notification handler code here
	AuxFunctionOperate(DH_EXTPTZ_LIGHTCONTROL, true);
}

void CRealPlayAndPTZControlDlg::OnBnClickedButtonRainBrushOpen()
{
	// TODO: Add your control notification handler code here
	AuxFunctionOperate(DH_PTZ_LAMP_CONTROL, false);
}

void CRealPlayAndPTZControlDlg::OnBnClickedButtonRainBrushClose()
{
	// TODO: Add your control notification handler code here
	AuxFunctionOperate(DH_PTZ_LAMP_CONTROL, true);
}

BOOL CRealPlayAndPTZControlDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	if (pMsg->message == WM_KEYDOWN || pMsg->message == WM_KEYUP)
	{
		if (VK_ESCAPE == pMsg->wParam || VK_RETURN == pMsg->wParam)
		{
			return TRUE;
		}
	}
	return CDialog::PreTranslateMessage(pMsg);
}

CRealPlayAndPTZControlDlg::~CRealPlayAndPTZControlDlg() {
	// notify the child thread
	m_is_alive = false;
	if(m_postprocessThread.joinable()) m_postprocessThread.join();
}

BOOL CRealPlayAndPTZControlDlg::DetectChessboard(CHESSBOARD_TYPE cb_type) {
	if (cb_type == CHESSBOARD_TYPE::CB_INSIDE) {
		m_cb_size = m_size_cb_in;
	}
	else if (cb_type == CHESSBOARD_TYPE::CB_OUTSIDE) {
		m_cb_size = m_size_cb_out;
	}
	else {
		m_logger.Error("chessboard size not supported!");
		return FALSE;
	}

	// list of points of corners
	//cv::resize(m_im_gray, m_im_gray_small, cv::Size(512, 288));
	m_im_gray_small = m_im_gray;
	m_cb_pts.resize(0);
	FindChessboardWrapper(this);

	if (!m_flag) {
		if(cb_type==CHESSBOARD_TYPE::CB_INSIDE)
			m_logger.Debug("cannot find chessboard inside!");
		else
			m_logger.Debug("cannot find chessboard outside!");
		return FALSE;
	}
	
	// rescale back to its original size
	float scale_x = m_im_gray.cols * 1.0 / m_im_gray_small.cols;
	float scale_y = m_im_gray.rows * 1.0 / m_im_gray_small.rows;
	for (int i = 0; i < m_cb_pts.size(); ++i) {
		m_cb_pts[i].x *= scale_x;
		m_cb_pts[i].y *= scale_y;
	}

	// get corners in sub-pixel
	cv::cornerSubPix(m_im_gray, m_cb_pts, cv::Size(11, 11), cv::Size(-1, -1), m_criteria);
	{
		// Log the data for later use
		std::stringstream str_buf;
		str_buf.clear();
		str_buf.str("");
		if (cb_type == CHESSBOARD_TYPE::CB_INSIDE) str_buf << "CB_INSIDE: ";
		else str_buf << "CB_OUTSIDE: ";
			
		for (auto pt_ : m_cb_pts) {
			str_buf << "(" << pt_.x << ", " << pt_.y << ")";
		}
		m_logger.Debug(str_buf.str().c_str());
	}
		
	// estimate the distance according to average height of the grid cell
	float cell_h;
	m_vec_cell_h.resize(0);

	for (size_t iRow = 1; iRow < m_cb_size.height; ++iRow) {
		for (size_t iCol = 0; iCol < m_cb_size.width; ++iCol) {
			float dx_sqr = std::powf(m_cb_pts[iRow * m_cb_size.width + iCol].x - \
				m_cb_pts[(iRow - 1) * m_cb_size.width + iCol].x, 2);
			float dy_sqr = std::powf(m_cb_pts[iRow * m_cb_size.width + iCol].y - \
				m_cb_pts[(iRow - 1) * m_cb_size.width + iCol].y, 2);
			cell_h = std::sqrt(dx_sqr + dy_sqr);
			m_vec_cell_h.push_back(cell_h);
		}
	}

	std::sort(m_vec_cell_h.begin(), m_vec_cell_h.end());
		
	// Get median value
	int n_all = (m_cb_size.height - 1) * m_cb_size.width;
	if (n_all % 2) cell_h = m_vec_cell_h[n_all / 2];
	else cell_h = 0.5 * (m_vec_cell_h[n_all / 2] + m_vec_cell_h[n_all - 1]);
		
	// log the grid cell information
	{
		std::stringstream ss("");
		ss << "cell_h=" << cell_h;
		m_logger.Debug(ss.str().c_str());
	}

	// reverse the points if required
	if (m_cb_pts[m_cb_size.width].y - m_cb_pts[0].y < 0) {
		std::reverse(m_cb_pts.begin(), m_cb_pts.end());
	}

	// calculate the distance according to ratio
	ASSERT(cell_h);
	float dist = (1.39 * 30.368) / cell_h;

	{
		std::stringstream ss("");
		ss << "dist=" << dist;
		m_logger.Debug(ss.str().c_str());
	}

	// Get the location of chessboard
	float sum_x, sum_y;
	sum_x = sum_y = 0;
	
	for (auto pt_ : m_cb_pts) {
		sum_x += pt_.x;
		sum_y += pt_.y;
	}

	n_all += m_cb_size.width;
	sum_x /= n_all;
	sum_y /= n_all;

	m_cb_center.x = int(sum_x);
	m_cb_center.y = int(sum_y);

	// draw the corners on the image
	cv::drawChessboardCorners(m_im_ori, m_cb_size, m_cb_pts, true);
	cv::circle(m_im_ori, m_cb_center, 10, cv::Scalar(0, 0, 255, 0), 3);

	// draw the distance number on image
	char buf_print[8];
	sprintf_s(buf_print, "%2.03fm", dist);
	cv::putText(m_im_ori, cv::String(buf_print), cv::Point(100, 100),
		cv::FONT_HERSHEY_COMPLEX, 2.0, cv::Scalar(30, 100, 250), 2);
	if (cb_type == CHESSBOARD_TYPE::CB_INSIDE) m_dist_inside = dist;
	else m_dist_outside = dist;

	return TRUE;
}


DWORD WINAPI FindChessboardWrapper(LPVOID _ptr) {
	CRealPlayAndPTZControlDlg* _this = (CRealPlayAndPTZControlDlg*)_ptr;
	_this->m_flag = cv::findChessboardCorners(
		_this->m_im_gray_small, 
		_this->m_cb_size, _this->m_cb_pts,
		cv::CALIB_CB_ADAPTIVE_THRESH + 
		cv::CALIB_CB_FAST_CHECK + 
		cv::CALIB_CB_NORMALIZE_IMAGE,
		0);
	return 0;
}

void CRealPlayAndPTZControlDlg::FollowTarget() {
	if(m_cb_is_found_inside){
		m_missing_counter = 0;
		// rotate the camera to move chessboard to the center of the image
		cv::Point center_(m_rect_ori.width / 2, m_rect_ori.height / 2);
		int dist_x_percent = 100*std::abs(m_cb_center.x - center_.x) / center_.x;
		char buf[32];
		sprintf_s(buf, "dist_x: %d%%", dist_x_percent);
		m_logger.Debug(buf);
	
		if (dist_x_percent > m_follow_threshold_percent) {
			if (m_cb_center.x - center_.x < 0) {
				// target is on the left, so move to left to align center
				if (!m_is_moving_left) {
					PtzControl(DH_PTZ_LEFT_CONTROL, FALSE);
					m_is_moving_left = true;
					m_is_moving_right = false;
				}
			}
			else {
				// target is on the right, so move to right to align center
				if (!m_is_moving_right) {
					PtzControl(DH_PTZ_RIGHT_CONTROL, FALSE);
					m_is_moving_right = true;
					m_is_moving_left = false;
				}
			}
		}
		else {
			if (m_is_moving_left) {
				PtzControl(DH_PTZ_LEFT_CONTROL, TRUE);
				m_is_moving_left = false;
				m_is_moving_right = false;
			}
			if (m_is_moving_right) {
				PtzControl(DH_PTZ_RIGHT_CONTROL, TRUE);
				m_is_moving_right = false;
				m_is_moving_left = false;
			}
		}
	}
	else {
		// chessboard not found
		m_missing_counter++;
		// if this state lasts for long, then circle around to look for it
		if (m_missing_counter == m_missing_limit) {
			m_logger.Debug("Start searching left...");
			// clear the tracking state and rotate the camera leftwards 
			m_is_moving_left = false;
			m_is_moving_right = false;
			PtzControl(DH_PTZ_LEFT_CONTROL, FALSE);
		}
		else if (m_missing_counter == m_missing_limit + m_reverse_limit) {
			m_logger.Debug("Start searching right...");
			// rotate the camera rightwards
			PtzControl(DH_PTZ_RIGHT_CONTROL, FALSE);
		} else if (m_missing_counter == m_missing_limit + 2*m_reverse_limit){
			// reset counter to reverse searching direction again
			m_missing_counter = m_missing_limit - 1;
		}
	}
}

