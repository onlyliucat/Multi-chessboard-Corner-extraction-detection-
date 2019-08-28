/*  Copyright 2017 onlyliu(997737609@qq.com).                             */
/*                                                                        */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */

#pragma once
               
#include "opencv2/opencv.hpp"
#include <vector>
#include "HeaderCB.h"

class ChessboradStruct
{
public:
	ChessboradStruct();
	~ChessboradStruct();

	cv::Mat initChessboard(Corners& corners, int idx);
	void chessboardsFromCorners(Corners& corners, std::vector<cv::Mat>& chessboards, float lamda = 0.5);
	int directionalNeighbor(int idx, cv::Vec2f v, cv::Mat chessboard, Corners& corners, int& neighbor_idx, float& min_dist);
	float chessboardEnergy(cv::Mat chessboard, Corners& corners);
	void predictCorners(std::vector<cv::Vec2f>& p1, std::vector<cv::Vec2f>& p2, std::vector<cv::Vec2f>& p3, std::vector<cv::Vec2f>& pred);
	cv::Mat growChessboard(cv::Mat chessboard, Corners& corners, int border_type);
	void assignClosestCorners(std::vector<cv::Vec2f>&cand, std::vector<cv::Vec2f>&pred, std::vector<int> &idx);
	void drawchessboard(cv::Mat img, Corners& corners, std::vector<cv::Mat>& chessboards, char * title = "chessboard", int t = 0, cv::Rect rect = cv::Rect(0,0,0,0));
	
	cv::Mat chessboard;
	float m_lamda;

};

