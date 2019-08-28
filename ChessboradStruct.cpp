/*  Copyright 2017 onlyliu(997737609@qq.com).                             */
/*                                                                        */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */

#include "ChessboradStruct.h"
#include <fstream>  
#include <limits>
#include<numeric>

#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */

ChessboradStruct::ChessboradStruct()
{

}

ChessboradStruct::~ChessboradStruct()
{

}

inline float distv(cv::Vec2f a, cv::Vec2f b)
{
	return std::sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]));
}

inline float mean_l(std::vector<float> &resultSet)
{
	double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	double mean = sum / resultSet.size(); //ŸùÖµ  
	return mean;
}

inline float stdev_l(std::vector<float> &resultSet, float &mean)
{
	double accum = 0.0;
	mean = mean_l(resultSet);
	std::for_each(std::begin(resultSet), std::end(resultSet), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	double stdev = sqrt(accum / (resultSet.size() - 1)); //·œ²î 
	return stdev;
}

inline float stdevmean(std::vector<float> &resultSet)
{
	float stdvalue, meanvalue;

	stdvalue = stdev_l(resultSet, meanvalue);

	return stdvalue / meanvalue;
}

int ChessboradStruct::directionalNeighbor(int idx, cv::Vec2f v, cv::Mat chessboard, Corners& corners, int& neighbor_idx, float& min_dist)
{

#if 1
	// list of neighboring elements, which are currently not in use
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;
			}
		}

	int nsize = unused.size();

	for (int i = 0; i < nsize;)
	{
		if (unused[i] < 0)
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0;
			nsize = unused.size();
			continue;
		}
		i++;
	}

	std::vector<float> dist_edge;
	std::vector<float> dist_point;

	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	for (int i = 0; i < unused.size(); i++)
	{
		int ind = unused[i];
		cv::Vec2f diri = cv::Vec2f(corners.p[ind].x, corners.p[ind].y) - idxp;
		float disti = diri[0] * v[0] + diri[1] * v[1];

		cv::Vec2f de = diri - disti*v;
		dist_edge.push_back(distv(de, cv::Vec2f(0, 0)));
		// distances
		dist_point.push_back(disti);
	}
#else
	// list of neighboring elements, which are currently not in use
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;//flag the used idx
			}
		}

	std::vector<float> dist_edge;
	std::vector<float> dist_point;

	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	for (int i = 0; i < corners.p.size(); i++)
	{
		if (unused[i] == -1)
		{
			dist_point.push_back(std::numeric_limits<float>::max());
			dist_edge.push_back(0);
			continue;
		}
		cv::Vec2f diri = cv::Vec2f(corners.p[i].x, corners.p[i].y) - idxp;
		float disti = diri[0] * v[0] + diri[1] * v[1];

		cv::Vec2f de = diri - disti*v;
		dist_edge.push_back(distv(de, cv::Vec2f(0, 0)));
		// distances
		dist_point.push_back(disti);
	}

#endif

	// find best neighbor
	int min_idx = 0;
	min_dist = std::numeric_limits<float>::max();

	//min_dist = dist_point[0] + 5 * dist_edge[0];
	for (int i = 0; i < dist_point.size(); i++)
	{
		if (dist_point[i] > 0)
		{
			float m = dist_point[i] + 5 * dist_edge[i];
			if (m < min_dist)
			{
				min_dist = m;
				min_idx = i;
			}
		}
	}
	neighbor_idx = unused[min_idx];

	return 1;
}


cv::Mat ChessboradStruct::initChessboard(Corners& corners, int idx)
{
	// return if not enough corners
	if (corners.p.size() < 9)
	{
		logd("not enough corners!\n");
		chessboard.release();//return empty!
		return chessboard;
	}
	// init chessboard hypothesis
	chessboard = -1 * cv::Mat::ones(3, 3, CV_32S);
	
	// extract feature index and orientation(central element)
	cv::Vec2f v1 = corners.v1[idx];
	cv::Vec2f v2 = corners.v2[idx];
	chessboard.at<int>(1, 1) = idx;
	std::vector<float> dist1(2), dist2(6);

	// find left / right / top / bottom neighbors
	directionalNeighbor(idx, +1 * v1, chessboard, corners, chessboard.at<int>(1, 2), dist1[0]);
	directionalNeighbor(idx, -1 * v1, chessboard, corners, chessboard.at<int>(1, 0), dist1[1]);
	directionalNeighbor(idx, +1 * v2, chessboard, corners, chessboard.at<int>(2, 1), dist2[0]);
	directionalNeighbor(idx, -1 * v2, chessboard, corners, chessboard.at<int>(0, 1), dist2[1]);

	// find top - left / top - right / bottom - left / bottom - right neighbors
	
	directionalNeighbor(chessboard.at<int>(1, 0), -1 * v2, chessboard, corners, chessboard.at<int>(0, 0), dist2[2]);
	directionalNeighbor(chessboard.at<int>(1, 0), +1 * v2, chessboard, corners, chessboard.at<int>(2, 0), dist2[3]);
	directionalNeighbor(chessboard.at<int>(1, 2), -1 * v2, chessboard, corners, chessboard.at<int>(0, 2), dist2[4]);
	directionalNeighbor(chessboard.at<int>(1, 2), +1 * v2, chessboard, corners, chessboard.at<int>(2, 2), dist2[5]);

	// initialization must be homogenously distributed
		

		bool sigood = false;
		sigood = sigood||(dist1[0]<0) || (dist1[1]<0);
		sigood = sigood || (dist2[0]<0) || (dist2[1]<0) || (dist2[2]<0) || (dist2[3]<0) || (dist2[4]<0) || (dist2[5]<0);
		

		sigood = sigood || (stdevmean(dist1) > 0.3) || (stdevmean(dist2) > 0.3);

		if (sigood == true)
		{
			chessboard.release();
			return chessboard;
		}
		return chessboard;
}

float ChessboradStruct::chessboardEnergy(cv::Mat chessboard, Corners& corners)
{
         float lamda = m_lamda;
	//energy: number of corners
	float E_corners = -1 * chessboard.size().area();
	//energy: structur
	float E_structure = 0;
	//walk through rows
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols-2; j++)
		{
			std::vector<cv::Vec2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(i, k);
				x.push_back(corners.p[n]);
			}
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0,0));
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	//walk through columns
	for (int i = 0; i < chessboard.cols; i++)
		for (int j = 0; j < chessboard.rows-2; j++)
		{
			std::vector<cv::Vec2f> x;
			float E_structure0 = 0;
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(k, i);
				x.push_back(corners.p[n]);
			}
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0, 0));
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	// final energy
	float E = E_corners + lamda*chessboard.size().area()*E_structure;

	return E;
}

// replica prediction(new)
void ChessboradStruct::predictCorners(std::vector<cv::Vec2f>& p1, std::vector<cv::Vec2f>& p2, 
	std::vector<cv::Vec2f>& p3, std::vector<cv::Vec2f>& pred)
{
	cv::Vec2f v1, v2;
	float a1, a2, a3;
	float s1, s2, s3;
	pred.resize(p1.size());
	for (int i = 0; i < p1.size(); i++)
	{
		// compute vectors
		v1 = p2[i] - p1[i];
		v2 = p3[i] - p2[i];
		// predict angles
		a1 = atan2(v1[1], v1[0]);
		a2 = atan2(v1[1], v1[0]);
		a3 = 2.0 * a2 - a1;

		//predict scales
		s1 = distv(v1, cv::Vec2f(0, 0));
		s2 = distv(v2, cv::Vec2f(0, 0));
		s3 = 2 * s2 - s1;
		pred[i] = p3[i] + 0.75*s3*cv::Vec2f(cos(a3), sin(a3));
	}
}

void ChessboradStruct::assignClosestCorners(std::vector<cv::Vec2f>&cand, std::vector<cv::Vec2f>&pred, std::vector<int> &idx)
{
	//return error if not enough candidates are available
	if (cand.size() < pred.size())
	{
		idx.resize(1);
		idx[0] = -1;
		return;
	}
	idx.resize(pred.size());

	//build distance matrix
	cv::Mat D = cv::Mat::zeros(cand.size(), pred.size(), CV_32FC1);
	float mind = FLT_MAX;
	for (int i = 0; i < D.cols; i++)//ÁÐÓÅÏÈ
	{
		cv::Vec2f delta;
		for (int j = 0; j < D.rows; j++)
		{
			delta = cand[j] - pred[i];
			float s = distv(delta, cv::Vec2f(0, 0));
			D.at<float>(j, i) = s;
			if (s < mind)
			{
				mind = s;
			}
		}
	}
	
	// search greedily for closest corners
	for (int k = 0; k < pred.size(); k++)
	{
		bool isbreak = false;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (fabs(D.at<float>(i, j) - mind) < 10e-10)
				{
					idx[j] = i;
					for (int m = 0; m < D.cols; m++)
					{
						D.at<float>(i, m) = FLT_MAX;
					}
					for (int m = 0; m < D.rows; m++)
					{
						D.at<float>(m,j) = FLT_MAX;
					}
					isbreak = true;
					break;
				}
			}
			if (isbreak == true)
				break;
		}
		mind = FLT_MAX;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (D.at<float>(i, j) < mind)
				{
					mind = D.at<float>(i, j);
				}
			}
		}
	}
}



cv::Mat ChessboradStruct::growChessboard(cv::Mat chessboard, Corners& corners, int border_type)
{
	if (chessboard.empty() == true)
	{
		return chessboard;
	}
	std::vector<cv::Point2f> p = corners.p;
	// list of  unused feature elements
	std::vector<int> unused(p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;
			}
		}

	int nsize = unused.size();

	for (int i = 0; i < nsize; )
	{
		if (unused[i] < 0)
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0; 
			nsize = unused.size();
			continue;
		}
		i++;
	}

	// candidates from unused corners
	std::vector<cv::Vec2f> cand;
	for (int i = 0; i < unused.size(); i++)
	{
		cand.push_back(corners.p[unused[i]]);
	}
	// switch border type 1..4
	cv::Mat chesstemp;

	switch (border_type)
	{
	case 0:
	{
		std::vector<cv::Vec2f> p1, p2, p3,pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == chessboard.cols - 3)
				{				
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 0, 1, 0,0);

		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, chesstemp.cols - 1) = unused[idx[i]];//ÓÒ
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 1:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row == chessboard.rows - 3)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 2)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 1)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 1, 0, 0, 0, 0);
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(chesstemp.rows - 1, i) = unused[idx[i]];//ÏÂ
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 2:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == 2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}

		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 1, 0, 0, 0);//×ó
		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, 0) = unused[idx[i]];
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 3:
	{
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row ==  2)
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 1)
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 0)
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)
		{
			return chessboard;
		}
		cv::copyMakeBorder(chessboard, chesstemp, 1, 0, 0, 0, 0, 0);//ÉÏ
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(0, i) = unused[idx[i]];
		}
		chessboard = chesstemp.clone();
		break;
	}
	default:
		break;
	}
	return chessboard;
}




void ChessboradStruct::chessboardsFromCorners( Corners& corners, std::vector<cv::Mat>& chessboards, float lamda)
{
	logd("Structure recovery:\n");
         m_lamda =  lamda;
	for (int i = 0; i < corners.p.size(); i++)
	{
		if (i % 128 == 0)
			printf("%d, %d\n", i, corners.p.size());
	
		cv::Mat csbd = initChessboard(corners, i);
		if (csbd.empty() == true)
		{
			continue;
		}
		float E =chessboardEnergy(csbd, corners);
		if (E > 0){ continue; }		
		
		int s = 0;
		//try growing chessboard
		while (true)
		{
			s++;
			// compute current energy
			float energy = chessboardEnergy(chessboard, corners);

			std::vector<cv::Mat>  proposal(4);
			std::vector<float> p_energy(4);
			//compute proposals and energies
			for (int j = 0; j < 4; j++)
			{
				proposal[j] = growChessboard(chessboard, corners, j);
 				p_energy[j] = chessboardEnergy(proposal[j], corners);
			}
			// find best proposal
			float min_value = p_energy[0];
			int min_idx = 0;
			for (int i0 = 1; i0 < p_energy.size(); i0++)
			{
				if (min_value > p_energy[i0])
				{
					min_value = p_energy[i0];
					min_idx = i0;
				}
			}
			// accept best proposal, if energy is reduced
			cv::Mat chessboardt;
			if (p_energy[min_idx] < energy)
			{
				chessboardt = proposal[min_idx];
				chessboard = chessboardt.clone();
			}
			else
			{
				break;
			}
		}//end while

		if (chessboardEnergy(chessboard, corners) < -10)
		{
			//check if new chessboard proposal overlaps with existing chessboards
			cv::Mat overlap = cv::Mat::zeros(cv::Size(2,chessboards.size()), CV_32FC1);
			for (int j = 0; j < chessboards.size(); j++)
			{
				bool isbreak = false;
				for (int k = 0; k < chessboards[j].size().area(); k++)
				{
					int refv = chessboards[j].at<int>(k / chessboards[j].cols, k%chessboards[j].cols);
					for (int l = 0; l < chessboard.size().area(); l++)
					{
						int isv = chessboard.at<int>(l/ chessboard.cols, l%chessboard.cols);
						if (refv == isv)
						{
							overlap.at<float>(j, 0) = 1.0;
							float s = chessboardEnergy(chessboards[j], corners);
							overlap.at<float>(j, 1) = s;
							isbreak = true;
							break;
						}
					}
				//	if (isbreak == true)
				//	{
					//	break;
				//	}
				}
				//if (isbreak == true)
				//{
				//	break;
				//}
			}//endfor

			// add chessboard(and replace overlapping if neccessary)
			bool isoverlap = false;
			for (int i0 = 0; i0 < overlap.rows; i0++)
			{
				if (overlap.empty() == false)
				{
					if (fabs(overlap.at<float>(i0, 0)) > 0.000001)// ==1
					{
						isoverlap = true;
						break;
					}
				}
			}
			if (isoverlap == false)
			{
				chessboards.push_back(chessboard);
			}
			else
			{
				bool flagpush = true;
				std::vector<bool> flagerase(overlap.rows);
				for (int m = 0; m < flagerase.size(); m++)
				{
					flagerase[m] = false;
				}
				float ce = chessboardEnergy(chessboard, corners);
				for (int i1 = 0; i1 < overlap.rows; i1++)
				{
					if (fabs(overlap.at<float>(i1, 0)) > 0.0001)// ==1//ÓÐÖØµþ
					{	
						bool isb1 = overlap.at<float>(i1, 1) > ce;

						int a = int(overlap.at<float>(i1, 1) * 1000);
						int b = int(ce * 1000);

						bool isb2 = a > b;
						if (isb1 != isb2)
							printf("find bug!\n");

						if (isb2)
						{	
							flagerase[i1] = true;
						}
						else
						{
							flagpush = false;
						//	break;
							
						}//endif
					}//endif
			
				}//end for

				if (flagpush == true)
				{
					for (int i1 = 0; i1 < chessboards.size();)
					{
						std::vector<cv::Mat>::iterator it = chessboards.begin() + i1;
						std::vector<bool>::iterator it1 = flagerase.begin() + i1;
						if (*it1  == true)
						{
							chessboards.erase(it);
							flagerase.erase(it1);
							i1 = 0;
						}
						i1++;
					}
					chessboards.push_back(chessboard);
				}

			}//endif
				
		}//endif
	}//end for 

}
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
void ChessboradStruct::drawchessboard(cv::Mat img, Corners& corners, std::vector<cv::Mat>& chessboards, char * title, int t_, cv::Rect rect)
{
        printf("end!\n");
        
	cv::RNG rng(0xFFFFFFFF);
	std::string s("If it's useful, please give a star ^-^.");
	std::string s1("https://github.com/onlyliucat\n");
	std::cout<<BOLDBLUE<<s<<std::endl<<BOLDGREEN<<s1<<std::endl;
	cv::Mat disp = img.clone();

	if (disp.channels() < 3)
		cv::cvtColor(disp, disp, CV_GRAY2BGR);
	float scale = 0.3;
	int n = 8;
	if (img.rows < 2000 || img.cols < 2000)
	{
		scale = 1;
		n = 2;
	}
	for (int k = 0; k < chessboards.size(); k++)
	{
		cv::Scalar s(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
		s = s * 255;
	
		for (int i = 0; i < chessboards[k].rows; i++)
			for (int j = 0; j < chessboards[k].cols; j++)
			{
				int d = chessboards[k].at<int>(i, j);
				cv::circle(disp, cv::Point2f(corners.p[d].x + rect.x, corners.p[d].y + rect.y), n, s, n);
			}
	}
	cv::Mat SmallMat;
	cv::resize(disp, SmallMat, cv::Size(), scale, scale);
	cv::namedWindow(title);
	cv::imshow(title, SmallMat);
	cv::waitKey(t_);
}



