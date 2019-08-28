/*  Copyright 2017 onlyliu(997737609@qq.com).                                */
/*                                                                        */
/*  part of source code come from https://github.com/qibao77/cornerDetect */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */


#include "CornerDetAC.h"
#include "corealgmatlab.h"


using namespace cv;
using namespace std;

//#define show_ 

CornerDetAC::CornerDetAC()
{
}


CornerDetAC::~CornerDetAC()
{
}
CornerDetAC::CornerDetAC(cv::Mat img)
{
	//3 scales
	radius.push_back(4);
	radius.push_back(8);
	radius.push_back(12);

	templateProps.push_back(Point2f((dtype)0, (dtype)CV_PI / 2));
	templateProps.push_back(Point2f((dtype)CV_PI / 4, (dtype)-CV_PI / 4));
	templateProps.push_back(Point2f((dtype)0, (dtype)CV_PI / 2));
	templateProps.push_back(Point2f((dtype)CV_PI / 4, (dtype)-CV_PI / 4));
	templateProps.push_back(Point2f((dtype)0, (dtype)CV_PI / 2));
	templateProps.push_back(Point2f((dtype)CV_PI / 4, (dtype)-CV_PI / 4));
}

//Normal probability density function (pdf).
dtype CornerDetAC::normpdf(dtype dist, dtype mu, dtype sigma)
{
	dtype s = exp(-0.5*(dist - mu)*(dist - mu) / (sigma*sigma));
	s = s / (std::sqrt(2 * CV_PI)*sigma);
	return s;
}


//**************************生成核*****************************//
//angle代表核类型：45度核和90度核
//kernelSize代表核大小（最终生成的核的大小为kernelSize*2+1）
//kernelA...kernelD是生成的核
//*************************************************************************//
void CornerDetAC::createkernel(float angle1, float angle2, int kernelSize, Mat &kernelA, Mat &kernelB, Mat &kernelC, Mat &kernelD)
{

	int width = (int)kernelSize * 2 + 1;
	int height = (int)kernelSize * 2 + 1;
	kernelA = cv::Mat::zeros(height, width, mtype);
	kernelB = cv::Mat::zeros(height, width, mtype);
	kernelC = cv::Mat::zeros(height, width, mtype);
	kernelD = cv::Mat::zeros(height, width, mtype);

	for (int u = 0; u < width; ++u){
		for (int v = 0; v < height; ++v){
			dtype vec[] = { u - kernelSize, v - kernelSize };//相当于将坐标原点移动到核中心
			dtype dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);//相当于计算到中心的距离
			dtype side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1);//相当于将坐标原点移动后的核进行旋转，以此产生四种核
			dtype side2 = vec[0] * (-sin(angle2)) + vec[1] * cos(angle2);//X=X0*cos+Y0*sin;Y=Y0*cos-X0*sin
			if (side1 <= -0.1&&side2 <= -0.1){
				kernelA.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1&&side2 >= 0.1){
				kernelB.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 <= -0.1&&side2 >= 0.1){
				kernelC.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1&&side2 <= -0.1){
				kernelD.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
		}
	}
	//std::cout << "kernelA:" << kernelA << endl << "kernelB:" << kernelB << endl
	//	<< "kernelC:" << kernelC<< endl << "kernelD:" << kernelD << endl;
	//归一化
	kernelA = kernelA / cv::sum(kernelA)[0];
	kernelB = kernelB / cv::sum(kernelB)[0];
	kernelC = kernelC / cv::sum(kernelC)[0];
	kernelD = kernelD / cv::sum(kernelD)[0];

}

//**************************//获取最小值*****************************//
//*************************************************************************//
void CornerDetAC::getMin(Mat src1, Mat src2, Mat &dst){
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous()){
		nc = nc*nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i < nr; i++){
		const dtype* dataLeft = src1.ptr<dtype>(i);
		const dtype* dataRight = src2.ptr<dtype>(i);
		dtype* dataResult = dst.ptr<dtype>(i);
		for (int j = 0; j < nc*channels; ++j){
			dataResult[j] = (dataLeft[j] < dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}
//**************************//获取最大值*****************************//
//*************************************************************************//
void CornerDetAC::getMax(Mat src1, Mat src2, Mat &dst)
{
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous()){
		nc = nc*nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i < nr; i++){
		const dtype* dataLeft = src1.ptr<dtype>(i);
		const dtype* dataRight = src2.ptr<dtype>(i);
		dtype* dataResult = dst.ptr<dtype>(i);
		for (int j = 0; j < nc*channels; ++j){
			dataResult[j] = (dataLeft[j] >= dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}
//获取梯度角度和权重
void CornerDetAC::getImageAngleAndWeight(Mat img, Mat &imgDu, Mat &imgDv, Mat &imgAngle, Mat &imgWeight)
{
	
	Mat sobelKernel(3, 3, mtype);
	Mat sobelKernelTrs(3, 3, mtype);
	//soble滤波器算子核
	sobelKernel.col(0).setTo(cv::Scalar(-1.0));
	sobelKernel.col(1).setTo(cv::Scalar(0.0));
	sobelKernel.col(2).setTo(cv::Scalar(1.0));

	sobelKernelTrs = sobelKernel.t();

	imgDu = corealgmatlab::conv2(img, sobelKernel, CONVOLUTION_SAME);
	imgDv = corealgmatlab::conv2(img, sobelKernelTrs, CONVOLUTION_SAME);

	if (imgDu.size() != imgDv.size())return;

	cartToPolar(imgDu, imgDv, imgWeight, imgAngle, false);
	for (int i = 0; i < imgDu.rows; i++)
	{
		for (int j = 0; j < imgDu.cols; j++)
		{
			dtype* dataAngle = imgAngle.ptr<dtype>(i);
			if (dataAngle[j] < 0)
				dataAngle[j] = dataAngle[j] + CV_PI;
			else if (dataAngle[j] > CV_PI)
				dataAngle[j] = dataAngle[j] - CV_PI;
		}
	}
	/*
	for (int i = 0; i < imgDu.rows; i++)
	{
		dtype* dataDv = imgDv.ptr<dtype>(i);
		dtype* dataDu = imgDu.ptr<dtype>(i);
		dtype* dataAngle = imgAngle.ptr<dtype>(i);
		dtype* dataWeight = imgWeight.ptr<dtype>(i);
		for (int j = 0; j < imgDu.cols; j++)
		{
			if (dataDu[j] > 0.000001)
			{
				dataAngle[j] = atan2((dtype)dataDv[j], (dtype)dataDu[j]);
				if (dataAngle[j] < 0)dataAngle[j] = dataAngle[j] + CV_PI;
				else if (dataAngle[j] > CV_PI)dataAngle[j] = dataAngle[j] - CV_PI;
			}
			dataWeight[j] = std::sqrt((dtype)dataDv[j] * (dtype)dataDv[j] + (dtype)dataDu[j] * (dtype)dataDu[j]);
		}
	}
	*/
}
//**************************非极大值抑制*****************************//
//inputCorners是输入角点，outputCorners是非极大值抑制后的角点
//threshold是设定的阈值
//margin是进行非极大值抑制时检查方块与输入矩阵边界的距离，patchSize是该方块的大小
//*************************************************************************//
void CornerDetAC::nonMaximumSuppression(Mat& inputCorners, vector<Point2f>& outputCorners, int patchSize, dtype threshold, int margin)
{
	if (inputCorners.size <= 0)
	{
		cout << "The imput mat is empty!" << endl; return;
	}
	for (int i = margin + patchSize; i <= inputCorners.cols - (margin + patchSize+1); i = i + patchSize + 1)//移动检查方块，每次移动一个方块的大小
	{
		for (int j = margin + patchSize; j <= inputCorners.rows - (margin + patchSize+1); j = j + patchSize + 1)
		{
			dtype maxVal = inputCorners.ptr<dtype>(j)[i];
			int maxX = i; int maxY = j;
			for (int m = i; m <= i + patchSize ; m++)//找出该检查方块中的局部最大值
			{
				for (int n = j; n <= j + patchSize ; n++)
				{
					dtype temp = inputCorners.ptr<dtype>(n)[m];
					if (temp > maxVal)
					{
						maxVal = temp; maxX = m; maxY = n;
					}
				}
			}
			if (maxVal < threshold)continue;//若该局部最大值小于阈值则不满足要求
			int flag = 0;
			for (int m = maxX - patchSize; m <= min(maxX + patchSize, inputCorners.cols - margin-1); m++)//二次检查
			{
				for (int n = maxY - patchSize; n <= min(maxY + patchSize, inputCorners.rows - margin-1); n++)
				{
					if (inputCorners.ptr<dtype>(n)[m]>maxVal && (m<i || m>i + patchSize || n<j || n>j + patchSize))
					{
						flag = 1; break;
					}
				}
				if (flag)break;
			}
			if (flag)continue;
			outputCorners.push_back(Point(maxX, maxY));
			std::vector<dtype> e1(2, 0.0);
			std::vector<dtype> e2(2, 0.0);
			cornersEdge1.push_back(e1);
			cornersEdge2.push_back(e2);
		}
	}
}

int cmp(const pair<dtype, int> &a, const pair<dtype, int> &b)
{
	return a.first > b.first;
}

//find modes of smoothed histogram
void CornerDetAC::findModesMeanShift(vector<dtype> hist, vector<dtype> &hist_smoothed, vector<pair<dtype, int>> &modes, dtype sigma){
	//efficient mean - shift approximation by histogram smoothing
	//compute smoothed histogram
	bool allZeros = true;
	for (int i = 0; i < hist.size(); i++)
	{
		dtype sum = 0;
		for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); j++)
		{
			int idx = 0;
			idx = (i + j) % hist.size();
			sum = sum + hist[idx] * normpdf(j, 0, sigma);
		}
		hist_smoothed[i] = sum;
		if (abs(hist_smoothed[i] - hist_smoothed[0]) > 0.0001)allZeros = false;// check if at least one entry is non - zero
		//(otherwise mode finding may run infinitly)
	}
	if (allZeros)return;

	//mode finding
	for (int i = 0; i < hist.size(); i++)
	{
		int j = i;
		while (true)
		{
			float h0 = hist_smoothed[j];
			int j1 = (j + 1) % hist.size();
			int j2 = (j - 1) % hist.size();
			float h1 = hist_smoothed[j1];
			float h2 = hist_smoothed[j2];
			if (h1 >= h0 && h1 >= h2)
				j = j1;
			else if (h2 > h0 && h2 > h1)
				j = j2;
			else 
				break;
		}
		bool ys = true;
		if (modes.size() == 0)
		{
			ys = true;
		}
		else
		{
			for (int k = 0; k < modes.size(); k++)
			{
				if (modes[k].second == j)
				{
					ys = false;
					break;
				}
			}
		}
		if (ys == true)
		{
			modes.push_back(std::make_pair(hist_smoothed[j], j));
		}
	}
	std::sort(modes.begin(), modes.end(), cmp);
}

//estimate edge orientations
void CornerDetAC::edgeOrientations(Mat imgAngle, Mat imgWeight, int index){
	//number of bins (histogram parameter)
	int binNum = 32;

	//convert images to vectors
	if (imgAngle.size() != imgWeight.size())return;
	vector<dtype> vec_angle, vec_weight;
	for (int i = 0; i < imgAngle.cols; i++)
	{
		for (int j = 0; j < imgAngle.rows; j++)
		{
			// convert angles from .normals to directions
			float angle = imgAngle.ptr<dtype>(j)[i] + CV_PI / 2;
			angle = angle > CV_PI ? (angle - CV_PI) : angle;
			vec_angle.push_back(angle);

			vec_weight.push_back(imgWeight.ptr<dtype>(j)[i]);
		}
	}

	//create histogram
	dtype pin = (CV_PI / binNum);
	vector<dtype> angleHist(binNum, 0);
	for (int i = 0; i < vec_angle.size(); i++)
	{
		int bin = max(min((int)floor(vec_angle[i] / pin), binNum - 1), 0);
		angleHist[bin] = angleHist[bin] + vec_weight[i];
	}

	// find modes of smoothed histogram
	vector<dtype> hist_smoothed(angleHist);
	vector<std::pair<dtype, int> > modes;
	findModesMeanShift(angleHist, hist_smoothed, modes, 1);

	// if only one or no mode = > return invalid corner
	if (modes.size() <= 1)return;

	//compute orientation at modes and sort by angle
	float fo[2];
	fo[0] = modes[0].second*pin;
	fo[1] = modes[1].second*pin;
	dtype deltaAngle = 0;
	if (fo[0] > fo[1])
	{
		dtype t = fo[0];
		fo[0] = fo[1];
		fo[1] = t;
	}

	deltaAngle = MIN(fo[1] - fo[0], fo[0] - fo[1] + (dtype)CV_PI);
	// if angle too small => return invalid corner
	if (deltaAngle <= 0.3)return;

	//set statistics: orientations
	cornersEdge1[index][0] = cos(fo[0]);
	cornersEdge1[index][1] = sin(fo[0]);
	cornersEdge2[index][0] = cos(fo[1]);
	cornersEdge2[index][1] = sin(fo[1]);
}

float CornerDetAC::norm2d(cv::Point2f o)
{
	return sqrt(o.x*o.x + o.y*o.y);
}
//亚像素精度找角点
void CornerDetAC::refineCorners(vector<Point2f> &cornors, Mat imgDu, Mat imgDv, Mat imgAngle, Mat imgWeight, float radius){
	// image dimensions
	int width = imgDu.cols;
	int height = imgDu.rows;
	// for all corners do
	for (int i = 0; i < cornors.size(); i++)
	{
		//extract current corner location
		int cu = cornors[i].x;
		int cv = cornors[i].y;
		// estimate edge orientations
		int startX, startY, ROIwidth, ROIheight;
		startX = MAX(cu - radius, (dtype)0);
		startY = MAX(cv - radius, (dtype)0);
		ROIwidth = MIN(cu + radius + 1, (dtype)width - 1) - startX;
		ROIheight = MIN(cv + radius + 1, (dtype)height - 1) - startY;

		Mat roiAngle, roiWeight;
		roiAngle = imgAngle(Rect(startX, startY, ROIwidth, ROIheight));
		roiWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight));
		edgeOrientations(roiAngle, roiWeight, i);

		// continue, if invalid edge orientations
		if (cornersEdge1[i][0] == 0 && cornersEdge1[i][1] == 0 || cornersEdge2[i][0] == 0 && cornersEdge2[i][1] == 0)
			continue;
	//	continue;

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//% corner orientation refinement %
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		cv::Mat A1 = cv::Mat::zeros(cv::Size(2, 2), mtype);
		cv::Mat A2 = cv::Mat::zeros(cv::Size(2, 2), mtype);

		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// pixel orientation vector
				cv::Point2f o(imgDu.at<dtype>(v, u), imgDv.at<dtype>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				// robust refinement of orientation 1
				dtype t0 = abs(o.x*cornersEdge1[i][0] + o.y*cornersEdge1[i][1]);
				if (t0 < 0.25) // inlier ?
				{
					Mat addtion(1, 2, mtype);
					addtion.col(0).setTo(imgDu.at<dtype>(v, u));
					addtion.col(1).setTo(imgDv.at<dtype>(v, u));
					Mat addtionu = imgDu.at<dtype>(v, u)*addtion;
					Mat addtionv = imgDv.at<dtype>(v, u)*addtion;
					for (int j = 0; j < A1.cols; j++)
					{
						A1.at<dtype>(0, j) = A1.at<dtype>(0, j) + addtionu.at<dtype>(0, j);
						A1.at<dtype>(1, j) = A1.at<dtype>(1, j) + addtionv.at<dtype>(0, j);
					}
				}
				// robust refinement of orientation 2
				dtype t1 = abs(o.x*cornersEdge2[i][0] + o.y*cornersEdge2[i][1]);
				if (t1 < 0.25) // inlier ?
				{
					Mat addtion(1, 2, mtype);
					addtion.col(0).setTo(imgDu.at<dtype>(v, u));
					addtion.col(1).setTo(imgDv.at<dtype>(v, u));
					Mat addtionu = imgDu.at<dtype>(v, u)*addtion;
					Mat addtionv = imgDv.at<dtype>(v, u)*addtion;
					for (int j = 0; j < A2.cols; j++)
					{
						A2.at<dtype>(0, j) = A2.at<dtype>(0, j) + addtionu.at<dtype>(0, j);
						A2.at<dtype>(1, j) = A2.at<dtype>(1, j) + addtionv.at<dtype>(0, j);
					}
				}
			}//end for
		// set new corner orientation
		cv::Mat v1, foo1;
		cv::Mat v2, foo2;
		cv::eigen(A1, v1, foo1);
		cv::eigen(A2, v2, foo2);
		cornersEdge1[i][0] = -foo1.at<dtype>(1, 0);
		cornersEdge1[i][1] = -foo1.at<dtype>(1, 1);
		cornersEdge2[i][0] = -foo2.at<dtype>(1, 0);
		cornersEdge2[i][1] = -foo2.at<dtype>(1, 1);

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//%  corner location refinement  %
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		cv::Mat G = cv::Mat::zeros(cv::Size(2, 2), mtype);
		cv::Mat b = cv::Mat::zeros(cv::Size(1, 2), mtype);
		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// pixel orientation vector
				cv::Point2f o(imgDu.at<dtype>(v, u), imgDv.at<dtype>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				//robust subpixel corner estimation
				if (u != cu || v != cv)// % do not consider center pixel
				{
					//compute rel. position of pixel and distance to vectors
					cv::Point2f w(u - cu, v - cv);
					float wvv1 = w.x*cornersEdge1[i][0] + w.y*cornersEdge1[i][1];
					float wvv2 = w.x*cornersEdge2[i][0] + w.y*cornersEdge2[i][1];

					cv::Point2f wv1(wvv1 * cornersEdge1[i][0], wvv1 * cornersEdge1[i][1]);
					cv::Point2f wv2(wvv2 * cornersEdge2[i][0], wvv2 * cornersEdge2[i][1]);
					cv::Point2f vd1(w.x - wv1.x, w.y - wv1.y);
					cv::Point2f vd2(w.x - wv2.x, w.y - wv2.y);
					dtype d1 = norm2d(vd1), d2 = norm2d(vd2);
					//if pixel corresponds with either of the vectors / directions
					if ((d1 < 3) && abs(o.x*cornersEdge1[i][0] + o.y*cornersEdge1[i][1]) < 0.25 \
						|| (d2 < 3) && abs(o.x*cornersEdge2[i][0] + o.y*cornersEdge2[i][1]) < 0.25)
					{
						dtype du = imgDu.at<dtype>(v, u), dv = imgDv.at<dtype>(v, u);
						cv::Mat uvt = (Mat_<dtype>(2, 1) << u, v);
						cv::Mat H = (Mat_<dtype>(2, 2) << du*du, du*dv, dv*du, dv*dv);
						G = G + H;
						cv::Mat t = H*(uvt);
						b = b + t;
					}
				}	
			}//endfor
		//set new corner location if G has full rank
		Mat s, u, v;
		SVD::compute(G, s, u, v);
		int rank = 0;
		for (int k = 0; k < s.rows; k++)
		{
			if (s.at<dtype>(k, 0) > 0.0001 || s.at<dtype>(k, 0) < -0.0001)// not equal zero
			{
				rank++;
			}
		}
		if (rank == 2)
		{
			cv::Mat mp = G.inv()*b;
			cv::Point2f  corner_pos_new(mp.at<dtype>(0, 0), mp.at<dtype>(1, 0));
			//  % set corner to invalid, if position update is very large
			if (norm2d(cv::Point2f(corner_pos_new.x - cu, corner_pos_new.y - cv)) >= 4)
			{
				cornersEdge1[i][0] = 0;
				cornersEdge1[i][1] = 0;
				cornersEdge2[i][0] = 0;
				cornersEdge2[i][1] = 0;
			}
			else
			{
				cornors[i].x = mp.at<dtype>(0, 0);
				cornors[i].y = mp.at<dtype>(1, 0);
			}
		}
		else//otherwise: set corner to invalid
		{
			cornersEdge1[i][0] = 0;
			cornersEdge1[i][1] = 0;
			cornersEdge2[i][0] = 0;
			cornersEdge2[i][1] = 0;
		}
	}
}

//compute corner statistics
void CornerDetAC::cornerCorrelationScore(Mat img, Mat imgWeight, vector<Point2f> cornersEdge, float &score){
	//center
	int c[] = { imgWeight.cols / 2, imgWeight.cols / 2 };

	//compute gradient filter kernel(bandwith = 3 px)
	Mat img_filter = Mat::ones(imgWeight.size(), imgWeight.type());
	img_filter = img_filter*-1;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			Point2f p1 = Point2f(i - c[0], j - c[1]);
			Point2f p2 = Point2f(p1.x*cornersEdge[0].x*cornersEdge[0].x + p1.y*cornersEdge[0].x*cornersEdge[0].y,
				p1.x*cornersEdge[0].x*cornersEdge[0].y + p1.y*cornersEdge[0].y*cornersEdge[0].y);
			Point2f p3 = Point2f(p1.x*cornersEdge[1].x*cornersEdge[1].x + p1.y*cornersEdge[1].x*cornersEdge[1].y,
				p1.x*cornersEdge[1].x*cornersEdge[1].y + p1.y*cornersEdge[1].y*cornersEdge[1].y);
			float norm1 = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
			float norm2 = sqrt((p1.x - p3.x)*(p1.x - p3.x) + (p1.y - p3.y)*(p1.y - p3.y));
			if (norm1 <= 1.5 || norm2 <= 1.5)
			{
				img_filter.ptr<dtype>(j)[i] = 1;
			}
		}
	}

	//normalize
	Mat mean, std, mean1, std1;
	meanStdDev(imgWeight, mean, std);
	meanStdDev(img_filter, mean1, std1);
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			imgWeight.ptr<dtype>(j)[i] = (dtype)(imgWeight.ptr<dtype>(j)[i] - mean.ptr<double>(0)[0]) / (dtype)std.ptr<double>(0)[0];
			img_filter.ptr<dtype>(j)[i] = (dtype)(img_filter.ptr<dtype>(j)[i] - mean1.ptr<double>(0)[0]) / (dtype)std1.ptr<double>(0)[0];
		}
	}

	//convert into vectors
	vector<float> vec_filter, vec_weight;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			vec_filter.push_back(img_filter.ptr<dtype>(j)[i]);
			vec_weight.push_back(imgWeight.ptr<dtype>(j)[i]);
		}
	}

	//compute gradient score
	float sum = 0;
	for (int i = 0; i < vec_weight.size(); i++)
	{
		sum += vec_weight[i] * vec_filter[i];
	}
	sum = (dtype)sum / (dtype)(vec_weight.size() - 1);
	dtype score_gradient = sum >= 0 ? sum : 0;

	//create intensity filter kernel
	Mat kernelA, kernelB, kernelC, kernelD;
	createkernel(atan2(cornersEdge[0].y, cornersEdge[0].x), atan2(cornersEdge[1].y, cornersEdge[1].x), c[0], kernelA, kernelB, kernelC, kernelD);//1.1 产生四种核

	//checkerboard responses
	float a1, a2, b1, b2;
	a1 = kernelA.dot(img);
	a2 = kernelB.dot(img);
	b1 = kernelC.dot(img);
	b2 = kernelD.dot(img);

	float mu = (a1 + a2 + b1 + b2) / 4;

	float score_a = (a1 - mu) >= (a2 - mu) ? (a2 - mu) : (a1 - mu);
	float score_b = (mu - b1) >= (mu - b2) ? (mu - b2) : (mu - b1);
	float score_1 = score_a >= score_b ? score_b : score_a;

	score_b = (b1 - mu) >= (b2 - mu) ? (b2 - mu) : (b1 - mu);
	score_a = (mu - a1) >= (mu - a2) ? (mu - a2) : (mu - a1);
	float score_2 = score_a >= score_b ? score_b : score_a;

	float score_intensity = score_1 >= score_2 ? score_1 : score_2;
	score_intensity = score_intensity > 0.0 ? score_intensity : 0.0;

	score = score_gradient*score_intensity;
}
//score corners
void CornerDetAC::scoreCorners(Mat img, Mat imgAngle, Mat imgWeight, vector<Point2f> &cornors, vector<int> radius, vector<float> &score){
	//for all corners do
	for (int i = 0; i < cornors.size(); i++)
	{
		//corner location
		int u = cornors[i].x+0.5;
		int v = cornors[i].y+0.5;
		if (i == 278)
		{
			int aaa = 0;
		}
		//compute corner statistics @ radius 1
		vector<float> scores;
		for (int j = 0; j < radius.size(); j++)
		{
			scores.push_back(0);
			int r = radius[j];
			if (u > r&&u <= (img.cols - r - 1) && v>r && v <= (img.rows - r - 1))
			{
				int startX, startY, ROIwidth, ROIheight;
				startX = u - r;
				startY = v - r;
				ROIwidth = 2 * r + 1;
				ROIheight = 2 * r + 1;

				Mat sub_img = img(Rect(startX, startY, ROIwidth, ROIheight)).clone();
				Mat sub_imgWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight)).clone();
				vector<Point2f> cornersEdge;
				cornersEdge.push_back(Point2f((float)cornersEdge1[i][0], (float)cornersEdge1[i][1]));
				cornersEdge.push_back(Point2f((float)cornersEdge2[i][0], (float)cornersEdge2[i][1]));
				cornerCorrelationScore(sub_img, sub_imgWeight, cornersEdge, scores[j]);
			}
		}
		//take highest score
		score.push_back(*max_element(begin(scores), end(scores)));
	}

}


void CornerDetAC::detectCorners(Mat &Src, vector<Point> &resultCornors, Corners& mcorners,  dtype scoreThreshold, bool isrefine)
{
	Mat gray, imageNorm;
	gray = Mat(Src.size(), CV_8U);

	// convert to double grayscale image
	if (Src.channels() == 3)
	{
		cvtColor(Src, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = Src.clone();
	}

	cv::GaussianBlur(gray, gray, cv::Size(9,9), 1.5);


	// scale input image
	normalize(gray, imageNorm, 0, 1, cv::NORM_MINMAX, mtype);
	//gray.convertTo(imageNorm, CV_32F, 1 / 255.0);

	// filter image
	Mat imgCorners = Mat::zeros(imageNorm.size(), mtype);

	Mat imgCornerA1(imageNorm.size(), mtype);
	Mat imgCornerB1(imageNorm.size(), mtype);
	Mat imgCornerC1(imageNorm.size(), mtype);
	Mat imgCornerD1(imageNorm.size(), mtype);

	Mat imgCornerA(imageNorm.size(), mtype);
	Mat imgCornerB(imageNorm.size(), mtype);
	Mat imgCorner1(imageNorm.size(), mtype);
	Mat imgCorner2(imageNorm.size(), mtype);
	Mat imgCornerMean(imageNorm.size(), mtype);

	std::cout << "begin filtering !" << std::endl;
	double t = (double)getTickCount();

	//#pragma omp parallel for num_threads(4)
	for (int i = 0; i < 6; i++)
	{
		Mat kernelA1, kernelB1, kernelC1, kernelD1;
		createkernel(templateProps[i].x, templateProps[i].y, radius[i / 2], kernelA1, kernelB1, kernelC1, kernelD1);//1.1 产生四种核

		//std::cout << "kernelA:" << kernelA1 << endl << "kernelB:" << kernelB1 << endl
		//	<< "kernelC:" << kernelC1 << endl << "kernelD:" << kernelD1 << endl;
		// filter image according with current template
#if 1
			imgCornerA1 = corealgmatlab::conv2(imageNorm, kernelA1, CONVOLUTION_SAME);
			imgCornerB1 = corealgmatlab::conv2(imageNorm, kernelB1, CONVOLUTION_SAME);
			imgCornerC1 = corealgmatlab::conv2(imageNorm, kernelC1, CONVOLUTION_SAME);
			imgCornerD1 = corealgmatlab::conv2(imageNorm, kernelD1, CONVOLUTION_SAME);
#else	
		filter2D(imageNorm, imgCornerA1, mtype, kernelA1);//a1
		filter2D(imageNorm, imgCornerB1, mtype, kernelB1);//a2
		filter2D(imageNorm, imgCornerC1, mtype, kernelC1);//b1
		filter2D(imageNorm, imgCornerD1, mtype, kernelD1);//b2
#endif	
		//compute mean
		imgCornerMean = (imgCornerA1 + imgCornerB1 + imgCornerC1 + imgCornerD1) / 4.0;//1.3 按照公式进行计算
		// case 1: a = white, b = black
		getMin(imgCornerA1 - imgCornerMean, imgCornerB1 - imgCornerMean, imgCornerA);
		getMin(imgCornerMean - imgCornerC1, imgCornerMean - imgCornerD1, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner1);
		// case 2: b = white, a = black
		getMin(imgCornerMean - imgCornerA1, imgCornerMean - imgCornerB1, imgCornerA);
		getMin(imgCornerC1 - imgCornerMean, imgCornerD1 - imgCornerMean, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner2);

		// update corner map
		getMax(imgCorners, imgCorner1, imgCorners);
		getMax(imgCorners, imgCorner2, imgCorners);
	}
#ifdef show_
	namedWindow("ROI");//创建窗口，显示原始图像
	imshow("ROI", imgCorners); waitKey(10);
#endif

	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "filtering time cost :" << t << std::endl;

	// extract corner candidates via non maximum suppression
	nonMaximumSuppression(imgCorners, cornerPoints, 3, 0.025, 5);//1.5 非极大值抑制算法进行过滤，获取棋盘格角点初步结果

	//post processing

	Mat imageDu(gray.size(), mtype);
	Mat imageDv(gray.size(), mtype);

	Mat img_angle = cv::Mat::zeros(gray.size(), mtype);
	Mat img_weight = cv::Mat::zeros(gray.size(), mtype);

	getImageAngleAndWeight(imageNorm, imageDu, imageDv, img_angle, img_weight);
	if (isrefine == true)
	{
		//subpixel refinement
		refineCorners(cornerPoints, imageDu, imageDv, img_angle, img_weight, 10);
		if (cornerPoints.size() > 0)
		{
			for (int i = 0; i < cornerPoints.size(); i++)
			{
				if (cornersEdge1[i][0] == 0 && cornersEdge1[i][0] == 0)
				{
					cornerPoints[i].x = 0; cornerPoints[i].y = 0;
				}
			}
		}
	}

	//remove corners without edges

	//score corners
	vector<float> score;
	scoreCorners(imageNorm, img_angle, img_weight, cornerPoints, radius, score);

#ifdef show_
	namedWindow("src");//创建窗口，显示原始图像
	imshow("src", Src); 
	waitKey(0);
#endif

	// remove low scoring corners
	int nlen = cornerPoints.size();
	if (nlen > 0)
	{
		for (int i = 0; i < nlen;i++)
		{
			if (score[i] > scoreThreshold)
			{
				mcorners.p.push_back(cornerPoints[i]);
				mcorners.v1.push_back(cv::Vec2f(cornersEdge1[i][0], cornersEdge1[i][1]));
				mcorners.v2.push_back(cv::Vec2f(cornersEdge2[i][0], cornersEdge2[i][1]));
				mcorners.score.push_back(score[i]);
			}
		}
	}

	std::vector<cv::Vec2f> corners_n1(mcorners.p.size());
	for (int i = 0; i < corners_n1.size(); i++)
	{
		if (mcorners.v1[i][0] + mcorners.v1[i][1] < 0.0)
		{
			mcorners.v1[i] = -mcorners.v1[i];
		}
		corners_n1[i] = mcorners.v1[i];
		float flipflag = corners_n1[i][0] * mcorners.v2[i][0] + corners_n1[0][1] * mcorners.v2[i][1];
		 if (flipflag > 0)
			 flipflag = -1;
		 else
			 flipflag = 1;
		 mcorners.v2[i] = flipflag * mcorners.v2[i];
	}
}

