// RealPlayAndPTZControlDlg.h : header file
//

#if !defined(AFX_RealPlayAndPTZControlDLG_H__475AE43D_B618_4F15_9DE9_2628AA826F52__INCLUDED_)
#define AFX_RealPlayAndPTZControlDLG_H__475AE43D_B618_4F15_9DE9_2628AA826F52__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "include/dhnetsdk.h"
#include "VideoNodeInfo.h"
#include "ExButton.h"
#include "ptzScreen.h"
#include <afxmt.h>
#include "PlayApi.h"
#include "Logger.h"
#include <opencv.hpp>
/////////////////////////////////////////////////////////////////////////////
// CRealPlayAndPTZControlDlg dialog

extern CPlayAPI g_PlayAPI;

class CRealPlayAndPTZControlDlg : public CDialog
{
// Construction
public:
	enum class CHESSBOARD_TYPE {
		CB_INSIDE = 0,
		CB_OUTSIDE
	};
	enum class ROTATE_MODE {
		RT_LEFT = 0,
		RT_RIGHT,
		RT_UP,
		RT_DOWN
	};
	Logger & m_logger;
	cv::Mat m_im_full, m_im_ori, m_vis_in, m_vis_out;
	cv::Size m_size_ori;
	cv::Rect m_rect_ori;
	CDC* m_canvas;
	CBitmap* m_bitmap;
	BOOL m_postprocess_initialized;
	cv::Rect m_rect_vis_in, m_rect_vis_out;
	cv::Mat m_im_gray;
	cv::Mat m_im_gray_small;

	CFont m_font;

	const cv::Size m_size_cb_in = cv::Size(5, 3);
	const cv::Size m_size_cb_out = cv::Size(5, 3);
	const cv::TermCriteria m_criteria = cv::TermCriteria(
		cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 3, 0.1);
	std::vector<cv::Point2f> m_cb_pts;
	std::vector<float> m_vec_cell_h;
	float m_dist_inside;
	float m_dist_outside;
	bool m_flag;
	cv::Size m_cb_size;
	cv::Point m_cb_center;

	bool m_cb_is_found_inside, m_cb_is_found_outside;
	const int m_follow_threshold_percent = 15;
	__volatile bool m_is_moving_left;
	__volatile bool m_is_moving_right;
	const int m_missing_limit = 100;
	const int m_reverse_limit = 1500;
	__volatile int m_missing_counter;
	__volatile bool m_is_alive;
	std::thread  m_postprocessThread;

	//PTZ extensive control function 
	void PtzExtControl(DWORD dwCommand, DWORD dwParam = 0);
	//PTZ control function 
	BOOL PtzControl(int type, BOOL stop);
	//Log in handle 
	LLONG m_LoginID;
	//Display function execution error reason.
	void LastError();
	//The callback interface
	void DeviceDisConnect(LLONG lLoginID, char *sDVRIP,LONG nDVRPort);
	//To get real-time data
	void ReceiveRealData(LLONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, LLONG lParam);
	//Callback function when device disconnected
	friend void CALLBACK DisConnectFunc(LLONG lLoginID, char *pchDVRIP, LONG nDVRPort, LDWORD dwUser);
	//Callback monitor data and then save
	friend void CALLBACK RealDataCallBackEx(LLONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, 
		DWORD dwBufSize, LLONG lParam, LDWORD dwUser);
	
	friend void CALLBACK  MessageProcFunc(int nWndID, UINT message, LDWORD dwUser);
	friend BOOL CALLBACK  GetParamsFunc(int nWndID, int type, LDWORD dwUser);
	friend void CALLBACK SetParamsFunc(int nWndID, int type, LDWORD dwUser);
	friend void CALLBACK RectEventFunc(RECT WinRect,CPoint &pointStart,CPoint &pointEnd,LDWORD dwUser);

	CRealPlayAndPTZControlDlg(Logger & logger, CWnd* pParent = NULL);	// standard constructor
	~CRealPlayAndPTZControlDlg();

	friend DWORD WINAPI FindChessboardWrapper(LPVOID _ptr);

	BOOL DetectChessboard(CHESSBOARD_TYPE cb_type);
	
	// Init func for postprocess handler
	BOOL InitPostprocessHandler();
	
	// postprocess frame handler
	BOOL Postprocess();

	// visualize the postprocessed result
	void VisualizeResult();

	// rotate the camera to follow target chessboard
	void FollowTarget();

	friend DWORD WINAPI MainProcedure(LPVOID _hDlg);
	
// Dialog Data
	//{{AFX_DATA(CRealPlayAndPTZControlDlg)
	enum { IDD = IDD_RealPlayAndPTZControl_DIALOG };
	CComboBox	m_comboPlayMode;
	CComboBox	m_auxNosel;
	CComboBox	m_comboPTZData;
	CExButton	m_iris_open;
	CExButton	m_iris_close;
	CExButton	m_focus_far;
	CExButton	m_focus_near;
	CExButton	m_zoom_tele;
	CExButton	m_zoom_wide;
	CExButton	m_ptz_rightdown;
	CExButton	m_ptz_rightup;
	CExButton	m_ptz_leftdown;
	CExButton	m_ptz_leftup;
	CExButton	m_ptz_right;
	CExButton	m_ptz_left;
	CExButton	m_ptz_down;
	CExButton	m_ptz_up;
	CComboBox	m_comboChannel;
	CComboBox	m_comboDispNum;
	CIPAddressCtrl	m_DvrIPAddr;
	CString	m_DvrUserName;
	CString	m_DvrPassword;
	DWORD	m_DvrPort;
	int		m_presetData; // [0, 100]
	int		m_crviseGroup;
	int		m_moveNo;
	int		m_posX;
	int		m_posY;
	int		m_posZoom;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CRealPlayAndPTZControlDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CRealPlayAndPTZControlDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnBTLogin();
	afx_msg void OnBUTTONPlay();
	afx_msg void OnBTLeave();
	afx_msg void OnDestroy();
	afx_msg void OnButtonStop();
	afx_msg void OnBtnPtzexctrl();
	afx_msg void OnPresetSet();
	afx_msg void OnPresetAdd();
	afx_msg void OnPresetDele();
	afx_msg void OnStartCruise();
	afx_msg void OnCruiseAddPoint();
	afx_msg void OnCruiseDelPoint();
	afx_msg void OnCruiseDelGroup();
	afx_msg void OnModeSetBegin();
	afx_msg void OnModeStart();
	afx_msg void OnModeSetDelete();
	afx_msg void OnLineSetLeft();
	afx_msg void OnLineSetRight();
	afx_msg void OnLineStart();
	afx_msg void OnFastGo();
	afx_msg void OnExactGo();
	afx_msg void OnResetZero();
	afx_msg void OnRotateStart();
	afx_msg void OnRotateStop();
	afx_msg void OnAuxOpen();
	afx_msg void OnAuxClose();
	afx_msg void OnBtnPtzmenu();
	afx_msg void OnSelchangeCOMBODispNum();
	afx_msg void OnCloseupCOMBODispNum();
	afx_msg LRESULT OnDisConnect(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnReConnect(WPARAM wParam, LPARAM lParam);
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
private:
	void MultiPlayServerMode(int iDispNum, HWND hWnd);
	void ShowLoginErrorReason(int nError);
	void ServerPlayMode(int iDispNum,int iChannel,HWND hWnd);
	void DirectPlayMode(int iDispNum,int iChannel,HWND hWnd);
	void MultiPlayMode(int iDispNum,HWND hWnd);
	BOOL SetMultiSpliteMode(int nChannel, int nMultiPlay);
	void StopPlayForMultiMode(int iDispNum);
	void StopPlayForServerMode(int iDispNum);
	void StopPlayForDirectMode(int iDispNum);
	void StopPlayForMultiServer(int iDispNum);
	
	void InitNetSDK();
	void InitPTZControl();
	void InitComboBox();
	HWND GetDispHandle(int nNum);
	CString GetDvrIP();
	void IsValid();
	long GetStreamPort(LLONG lRealHandle);
	void SetPlayVideoInfo(int iDispNum,int iChannel,enum RealPlayMode ePlayMode);
	void CloseDispVideo(int iDispNum);

	void UpdataScreenPos(void);
	void MessageProc(int nWndID, UINT message);
	BOOL GetParams(int nWndID, int type);
	void SetParams(int nWndID, int type);
	BOOL GetExitDecode(int nCurWndID);
	BOOL GetExitCycle(int nCurWndID);
	void SetExitDecode(int nCurWndID);
	void SetExitCycle(int nCurWndID);
	
	//Device channel amount 
	int m_nChannelCount;
	CRect m_rectSmall;
	CRect m_rectLarge;
	//9-window control information 
	CVideoNodeInfo m_videoNodeInfo[9];
	LLONG m_DispHanle[9];

	CPtzScreen m_ptzScreen;
	CRect			m_screenRect;
	CRect			m_clientRect;
	int				m_CurScreen;
	BOOL			m_bWndExitDecode[16];
	BOOL			m_bWndExitCycle[16];
	CCriticalSection m_cs;
	CCriticalSection m_csPos;
	
public:
	afx_msg void OnBnClickedButtonLightOpen();
	afx_msg void OnBnClickedButtonLightClose();
	afx_msg void OnBnClickedButtonRainBrushOpen();
	afx_msg void OnBnClickedButtonRainBrushClose();
	virtual BOOL PreTranslateMessage(MSG* pMsg);
private:
	template<typename T>
	void AuxFunctionOperate(T type, bool bStop)
	{
		UpdateData(TRUE);
		CString strDispNum;
		m_comboDispNum.GetWindowText(strDispNum);
		int iDispNum = atoi(strDispNum);
		int iChannel=m_videoNodeInfo[iDispNum-1].GetDvrChannel();
		if(-1 == iChannel)
		{
			return ;
		}
		
		BYTE param1 = 0, param2 = 0, param3 = 0;
		switch(type)
		{
		case DH_PTZ_LAMP_CONTROL:		
			param1 = bStop ? 0x00 : 0x01;			
			break;
		case DH_EXTPTZ_LIGHTCONTROL:			
			param3 = bStop ? 0x00 : 0x01;
			break;
		default:
			break;
		}

		BOOL bRet=CLIENT_DHPTZControlEx(m_LoginID, iChannel, type, param1, param2, param3, FALSE);
		if(bRet)
		{
			SetDlgItemText(IDC_PTZSTATUS, ConvertString("Succeed"));
		}
		else
		{
			DWORD dwError = CLIENT_GetLastError();
			SetDlgItemText(IDC_PTZSTATUS, ConvertString("Fail"));
		}
		return ;
	}
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_RealPlayAndPTZControlDLG_H__475AE43D_B618_4F15_9DE9_2628AA826F52__INCLUDED_)
