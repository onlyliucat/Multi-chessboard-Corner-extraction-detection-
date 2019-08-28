/*  Copyright 2017 onlyliu(997737609@qq.com).                             */
/*                                                                        */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */



#include "opencv2/opencv.hpp"

#include <algorithm>
#include "CornerDetAC.h"
#include "ChessboradStruct.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;
using std::vector;
vector<Point2i> points;

int main(int argc, char* argv[])
{
    Mat src1; 
    cv::Mat src;
    printf("read file...\n");
    string sr;
    if(argc < 2)
     sr = string("02.png");
    else
     sr = string(argv[1]);
    string simage, stxt, ssave;
    simage = sr ;//+ ".bmp";
    stxt = sr + ".txt";
    ssave = sr + ".png";
    ssave = "./t/"+ssave;
	src1 = imread(simage.c_str(), -1);//载入测试图像
	if (src1.channels() == 1)
	{
		src = src1.clone();
	}
	else
	{
		if (src1.channels() == 3)
		{
			cv::cvtColor(src1, src, CV_BGR2GRAY);
		}
		else
		{
			if (src1.channels() == 4)
			{
				cv::cvtColor(src1, src, CV_BGRA2GRAY);
			}
		}
	}

	if (src.empty())
	{
		printf("Cannot read image file: %s\n", simage.c_str());
		return -1;
	}
	else
	{
		printf("read image file ok\n");
	}

	vector<Point> corners_p;//存储找到的角点
	
	double t = (double)getTickCount();
	std::vector<cv::Mat> chessboards;
	CornerDetAC corner_detector(src);
	ChessboradStruct chessboardstruct;

	Corners corners_s;
	corner_detector.detectCorners(src, corners_p, corners_s, 0.01);

	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "time cost :" << t << std::endl;
     
	ImageChessesStruct ics;
        chessboardstruct.chessboardsFromCorners(corners_s, chessboards, 0.6);
        chessboardstruct.drawchessboard(src1, corners_s, chessboards, "cb", 0);

     return 0;
}

