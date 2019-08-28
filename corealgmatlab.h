#pragma once

#include "opencv2/opencv.hpp"
enum ConvolutionType {
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,

	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,
	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};

class corealgmatlab
{
public:
	corealgmatlab();
	~corealgmatlab();
	static 	cv::Mat conv2(const cv::Mat &img, const cv::Mat& ikernel, ConvolutionType type);
};

