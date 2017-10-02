/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Minimum Barrier Salient Object Detection at 80 FPS", Jianming Zhang, 
*	Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price, Radomir Mech, ICCV, 
*       2015
*	
*	Copyright (C) 2015 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: 
*       jimmie33@gmail.com
*******************************************************************************/

#include <iostream>
#include <ctime>
#include "opencv2/opencv.hpp"
//#include "mexopencv.hpp"
#include "MBS.hpp"
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

#define MAX_IMG_DIM 200
#define TOLERANCE 0.01
#define FRAME_MAX 20
#define SOBEL_THRESH 0.4
#define CLUSTER 4
bool flag[4];

int getThreshold(cv::Mat img, int width, int height)
{
	int size = width*height;
	cv::MatND outputhist;
	int hisSize[1] = { 256 };
	float range[2] = { 0.0, 255.0 };
	const float *ranges; ranges = &range[0];
	calcHist(&img, 1, 0, Mat(), outputhist, 1, hisSize, &ranges);
	double sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum = sum + i * outputhist.at<float>(i);
	}
	int threshold = 0;
	float sumvaluew = 0.0, sumvalue = 0.0, maximum = 0.0, wF, p12, diff, between;
	for (int i = 0; i < 256; i++)
	{
		sumvalue = sumvalue + outputhist.at<float>(i);
		sumvaluew = sumvaluew + i * outputhist.at<float>(i);
		wF = size - sumvalue;
		p12 = wF * sumvalue;
		if (p12 == 0){ p12 = 1; }
		diff = sumvaluew * wF - (sum - sumvaluew) * sumvalue;
		between = (float)diff * diff / p12;
		if (between >= maximum){
			threshold = i;
			maximum = between;
		}
	}
	return threshold;
}

cv::Rect getrectangular(cv::Mat img)
{
	IplImage pImg = IplImage(img);
	CvMemStorage * storage = cvCreateMemStorage(0);
	CvSeq * contour = 0, *contmax = 0;
	cvFindContours(&pImg, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	double area = 0, maxArea = 0;
	for (; contour; contour = contour->h_next)
	{
		area = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
		if (area > maxArea)
		{
			contmax = contour;
			maxArea = area;
		}
	}
	Rect maxRect = cvBoundingRect(contmax, 0);
	return maxRect;
}

MBS::MBS(const Mat& src)
:mAttMapCount(0)
{
	mSrc=src.clone();
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);

	split(mSrc, mFeatureMaps);

	for (int i = 0; i < mFeatureMaps.size(); i++)
	{
		//normalize(mFeatureMaps[i], mFeatureMaps[i], 255.0, 0.0, NORM_MINMAX);
		medianBlur(mFeatureMaps[i], mFeatureMaps[i], 5);
	}
}

void MBS::computeSaliency(bool use_geodesic)
{

        if (use_geodesic)
		mMBSMap = fastGeodesic(mFeatureMaps);
	else
		mMBSMap = fastMBS(mFeatureMaps);//得到Lab空间计算的salientmap
	normalize(mMBSMap, mMBSMap, 0.0, 1.0, NORM_MINMAX);//三通道叠加之后进行归一化处理
	mSaliencyMap = mMBSMap;
}

Mat MBS::getSaliencyMap()
{
	Mat ret;
	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

void rasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)
{
	Size sz = featMap.size();
	float *pMapup = (float*)map.data + 1;
	float *pMap = pMapup + sz.width;
	uchar *pFeatup = featMap.data + 1;
	uchar *pFeat = pFeatup + sz.width;
	uchar *pLBup = lb.data + 1;
	uchar *pLB = pLBup + sz.width;
	uchar *pUBup = ub.data + 1;
	uchar *pUB = pUBup + sz.width;

	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float lfV, upV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap - 1);
		featPrev = *(pFeat - 1);
		lbPrev = *(pLB - 1);
		ubPrev = *(pUB - 1);


		for (int c = 1; c < sz.width - 1; c++)
		{
			lfV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			upV = MAX(*pFeat, *pUBup) - MIN(*pFeat, *pLBup);//(*pFeat >= *pLBup && *pFeat <= *pUBup) ? *pMapup : *pMapup + abs((float)(*pFeat) - (float)(*pFeatup));

			flag = 0;
			if (lfV < *pMap)
			{
				*pMap = lfV;
				flag = 1;
			}
			if (upV < *pMap)
			{
				*pMap = upV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from left
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from up
				*pLB = MIN(*pFeat, *pLBup);
				*pUB = MAX(*pFeat, *pUBup);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap++; pMapup++;
			featPrev = *pFeat;
			pFeat++; pFeatup++;
			lbPrev = *pLB;
			pLB++; pLBup++;
			ubPrev = *pUB;
			pUB++; pUBup++;
		}
		pMapup += 2; pMap += 2;
		pFeat += 2; pFeatup += 2;
		pLBup += 2; pLB += 2;
		pUBup += 2; pUB += 2;
	}
}

void invRasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)
{
	Size sz = featMap.size();
	int datalen = sz.width*sz.height;
	float *pMapdn = (float*)map.data + datalen - 2;
	float *pMap = pMapdn - sz.width;
	uchar *pFeatdn = featMap.data + datalen - 2;
	uchar *pFeat = pFeatdn - sz.width;
	uchar *pLBdn = lb.data + datalen - 2;
	uchar *pLB = pLBdn - sz.width;
	uchar *pUBdn = ub.data + datalen - 2;
	uchar *pUB = pUBdn - sz.width;
	
	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float rtV, dnV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap + 1);
		featPrev = *(pFeat + 1);
		lbPrev = *(pLB + 1);
		ubPrev = *(pUB + 1);

		for (int c = 1; c < sz.width - 1; c++)
		{
			rtV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			dnV = MAX(*pFeat, *pUBdn) - MIN(*pFeat, *pLBdn);//(*pFeat >= *pLBdn && *pFeat <= *pUBdn) ? *pMapdn : *pMapdn + abs((float)(*pFeat) - (float)(*pFeatdn));

			flag = 0;
			if (rtV < *pMap)
			{
				*pMap = rtV;
				flag = 1;
			}
			if (dnV < *pMap)
			{
				*pMap = dnV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from right
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from down
				*pLB = MIN(*pFeat, *pLBdn);
				*pUB = MAX(*pFeat, *pUBdn);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap--; pMapdn--;
			featPrev = *pFeat;
			pFeat--; pFeatdn--;
			lbPrev = *pLB;
			pLB--; pLBdn--;
			ubPrev = *pUB;
			pUB--; pUBdn--;
		}


		pMapdn -= 2; pMap -= 2;
		pFeatdn -= 2; pFeat -= 2;
		pLBdn -= 2; pLB -= 2;
		pUBdn -= 2; pUB -= 2;
	}
}

cv::Mat fastMBS(const std::vector<cv::Mat> featureMaps)
{
	assert(featureMaps[0].type() == CV_8UC1);

	Size sz = featureMaps[0].size();
	Mat ret = Mat::zeros(sz, CV_32FC1);
	if (sz.width < 3 || sz.height < 3)
		return ret;

	for (int i = 0; i < featureMaps.size(); i++)//分别用LAB空间的三个通道计算得到salientmap
	{
		Mat map = Mat::zeros(sz, CV_32FC1);
		Mat mapROI(map, Rect(1, 1, sz.width - 2, sz.height - 2));
		mapROI.setTo(Scalar(100000));
		Mat lb = featureMaps[i].clone();
		Mat ub = featureMaps[i].clone();

		rasterScan(featureMaps[i], map, lb, ub);
		invRasterScan(featureMaps[i], map, lb, ub);
		rasterScan(featureMaps[i], map, lb, ub);
		ret += map;
	}

	return ret;
	
}

cv::Mat getBoundaryContrastMap(cv::Mat Src, int width, int height, int boundarysize)
{
	cv::imshow("Src", Src);
	waitKey();
	vector <Mat> BoundaryClusters(CLUSTER);
	BoundaryClusters[0] = Mat(height, boundarysize, Src.type(), Scalar(0));
	BoundaryClusters[1] = Mat(boundarysize, width, Src.type(), Scalar(0));
	BoundaryClusters[2] = Mat(boundarysize, width, Src.type(), Scalar(0));
	BoundaryClusters[3] = Mat(height, boundarysize, Src.type(), Scalar(0));
	vector <Rect> BoundaryRect(CLUSTER);
	BoundaryRect[0].x = BoundaryRect[0].y = BoundaryRect[1].x = BoundaryRect[1].y = BoundaryRect[2].x = BoundaryRect[3].y= 0;
	BoundaryRect[2].y = height - boundarysize; BoundaryRect[3].x = width - boundarysize;
	BoundaryRect[0].width = BoundaryRect[3].width = BoundaryRect[1].height = BoundaryRect[2].height = boundarysize;
	BoundaryRect[0].height = BoundaryRect[3].height = height; BoundaryRect[1].width = BoundaryRect[2].width = width;
	
	vector <Mat> covar(CLUSTER), meansmat(CLUSTER), data(CLUSTER); vector <bool> flags; double covinvert[4][3][3];
	for (int k = 0; k < CLUSTER; k++)
	{
		flag[k] = true;
		int num = BoundaryClusters[k].rows*BoundaryClusters[k].cols;
		int width = BoundaryClusters[k].cols;
		int height = BoundaryClusters[k].rows;
		data[k] = Mat(num, 3, CV_32FC1);
		BoundaryClusters[k] = Src(BoundaryRect[k]);
		//cv::imshow("BoundaryClusters", BoundaryClusters[3]);
		//waitKey();
		for (int  i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				data[k].at<float>(i*width + j, 0) = BoundaryClusters[k].data[(i*width + j) * 3 + 0];
				data[k].at<float>(i*width + j, 1) = BoundaryClusters[k].data[(i*width + j) * 3 + 1];
				data[k].at<float>(i*width + j, 2) = BoundaryClusters[k].data[(i*width + j) * 3 + 2];
			}
		}
		//data[k] = Mat(BoundaryClusters[k].rows*BoundaryClusters[k].cols, 1, CV_32FC3, BoundaryClusters[k].data);
		/*Scalar means1;
		means1 = cv::mean(data[k]);*/
		//Scalar means2 = cv::mean(data);
		calcCovarMatrix(data[k], covar[k], meansmat[k], CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
		if (determinant(covar[k]) == 0)
		{
			flag[k] = false;
			continue;
		}
		CvMat cvcovar = covar[k];
		if (cvDet(&cvcovar) == 0)//判断是否存在矩阵行列式为0，若存在则由马氏距离所求距离全部为0（矩阵奇异）；
			flag[k] = false;
		CvMat *covarinvert = cvCreateMat(3, 3, CV_32FC1);
		cvInvert(&cvcovar, covarinvert);
		for (int row = 0; row<covarinvert->height; row++)
		{
			float* pData = (float*)(covarinvert->data.ptr + row*covarinvert->step);
			for (int col = 0; col<covarinvert->width; col++)
			{
				covinvert[k][row][col] = *pData;
				pData++;
			}
		}
		//Mat covarinvert = Mat(3, 3, CV_32FC1);
		//covarinvert = covar[k].inv();
		/*for (int row = 0; row<3; row++)
		{
			for (int col = 0; col<3; col++)
			{
				covinvert[k][row][col] = covarinvert.data[row + col];
			}
		}*/

	}
	/*计算马氏距离*/
	float* MahalMap = (float *)malloc(height * width * sizeof(float));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j += 1)
		{
			float S[CLUSTER], submatrix[3], Smax = 0, total_distance = 0;
			int  total_num = 0;
			for (int g = 0; g < CLUSTER; g++)
			{
				if (!flag[g])
				{
					S[g] = 0;
					continue;
				}
				submatrix[0] = Src.data[(i*width + j) * 3 + 0] - meansmat[g].data[0];
				submatrix[1] = Src.data[(i*width + j) * 3 + 1] - meansmat[g].data[1];
				submatrix[2] = Src.data[(i*width + j) * 3 + 2] - meansmat[g].data[2];

				S[g] = sqrt(submatrix[0] * (submatrix[0] * covinvert[g][0][0] + submatrix[1] *covinvert[g][1][0] + submatrix[2] *covinvert[g][2][0]) +
					submatrix[1] * (submatrix[0] *covinvert[g][0][1] + submatrix[1] * covinvert[g][1][1] + submatrix[2] * covinvert[g][2][1]) +
					submatrix[2] * (submatrix[0] * covinvert[g][0][2] + submatrix[1] * covinvert[g][1][2] + submatrix[2] * covinvert[g][2][2]));
				/*if (S[g] > Smax)
				{
					Smax = S[g];
				}*/
				total_distance += BoundaryClusters[g].rows*BoundaryClusters[g].cols * S[g];
				total_num += BoundaryClusters[g].rows*BoundaryClusters[g].cols;
			}
			//MahalMap[i*width + j] = S[1];
			
			//MahalMap[i*width + j] = S[0] + S[1] + S[2] + S[3] - Smax;
			MahalMap[i*width + j] = total_distance / total_num;
		}
	}
	cv::Mat salient_map = Mat(height, width, CV_32FC1, MahalMap);
	//normalize(salient_map, salient_map, 0.0, 1.0, NORM_MINMAX);
	normalize(salient_map, salient_map, 255, 0, NORM_MINMAX);
	salient_map.convertTo(salient_map, CV_8UC1);
	cv::imshow("MahalMap", salient_map);
	waitKey();
	return salient_map;
}

float getThreshForGeo(const Mat& src)
{
	float ret = 0.00000;
	Size sz = src.size();

	uchar *pFeatup = src.data + 1;
	uchar *pFeat = pFeatup + sz.width;
	uchar *pfeatdn = pFeat + sz.width;

	float featPrev;

	for (int r = 1; r < sz.height - 1; r++)
	{
		featPrev = *(pFeat - 1);

		for (int c = 1; c < sz.width - 1; c++)
		{
			float temp = MIN(abs(*pFeat-featPrev),abs(*pFeat-*(pFeat+1)));
			temp = MIN(temp,abs(*pFeat-*pFeatup));
			temp = MIN(temp,abs(*pFeat-*pfeatdn));
			ret += temp;

			featPrev = *pFeat;
			pFeat++; pFeatup++; pfeatdn++;
		}
		pFeat += 2; pFeatup += 2; pfeatdn += 2;
	}
	return ret / ((sz.width - 2)*(sz.height - 2));
}

void rasterScanGeo(const Mat& featMap, Mat& map, float thresh)
{
	Size sz = featMap.size();
	float *pMapup = (float*)map.data + 1;
	float *pMap = pMapup + sz.width;
	uchar *pFeatup = featMap.data + 1;
	uchar *pFeat = pFeatup + sz.width;

	float mapPrev;
	float featPrev;

	float lfV, upV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap - 1);
		featPrev = *(pFeat - 1);


		for (int c = 1; c < sz.width - 1; c++)
		{
			lfV = (abs(featPrev - *pFeat)>thresh ? abs(featPrev - *pFeat):0.0f) + mapPrev;
			upV = (abs(*pFeatup - *pFeat)>thresh ? abs(*pFeatup - *pFeat):0.0f) + *pMapup;
			
			if (lfV < *pMap)
			{
				*pMap = lfV;
			}
			if (upV < *pMap)
			{
				*pMap = upV;
			}

			mapPrev = *pMap;
			pMap++; pMapup++;
			featPrev = *pFeat;
			pFeat++; pFeatup++;
		}
		pMapup += 2; pMap += 2;
		pFeat += 2; pFeatup += 2;
	}
}

void invRasterScanGeo(const Mat& featMap, Mat& map, float thresh)
{
	Size sz = featMap.size();
	int datalen = sz.width*sz.height;
	float *pMapdn = (float*)map.data + datalen - 2;
	float *pMap = pMapdn - sz.width;
	uchar *pFeatdn = featMap.data + datalen - 2;
	uchar *pFeat = pFeatdn - sz.width;

	float mapPrev;
	float featPrev;

	float rtV, dnV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap + 1);
		featPrev = *(pFeat + 1);

		for (int c = 1; c < sz.width - 1; c++)
		{
			rtV = (abs(featPrev - *pFeat)>thresh ? abs(featPrev - *pFeat):0.0f) + mapPrev;
			dnV = (abs(*pFeatdn - *pFeat)>thresh ? abs(*pFeatdn - *pFeat):0.0f) + *pMapdn;
			
			if (rtV < *pMap)
			{
				*pMap = rtV;
			}
			if (dnV < *pMap)
			{
				*pMap = dnV;
			}

			mapPrev = *pMap;
			pMap--; pMapdn--;
			featPrev = *pFeat;
			pFeat--; pFeatdn--;
		}


		pMapdn -= 2; pMap -= 2;
		pFeatdn -= 2; pFeat -= 2;
	}
}

cv::Mat fastGeodesic(const std::vector<cv::Mat> featureMaps)
{
	assert(featureMaps[0].type() == CV_8UC1);

	Size sz = featureMaps[0].size();
	Mat ret = Mat::zeros(sz, CV_32FC1);
	if (sz.width < 3 || sz.height < 3)
		return ret;


	for (int i = 0; i < featureMaps.size(); i++)
	{
		// determines the threshold for clipping
		float thresh = getThreshForGeo(featureMaps[i]);
		//cout << thresh << endl;
		Mat map = Mat::zeros(sz, CV_32FC1);
		Mat mapROI(map, Rect(1, 1, sz.width - 2, sz.height - 2));
		mapROI.setTo(Scalar(1000000000));

		rasterScanGeo(featureMaps[i], map, thresh);
		invRasterScanGeo(featureMaps[i], map, thresh);
		rasterScanGeo(featureMaps[i], map, thresh);

		ret += map;
	}

	return ret;

}

int findFrameMargin(const Mat& img, bool reverse)
{
	Mat edgeMap, edgeMapDil, edgeMask;
	Sobel(img, edgeMap, CV_16SC1, 0, 1);
	edgeMap = abs(edgeMap);
	edgeMap.convertTo(edgeMap, CV_8UC1);
	edgeMask = edgeMap < (SOBEL_THRESH * 255.0);
	dilate(edgeMap, edgeMapDil, Mat(), Point(-1, -1), 2);
	edgeMap = edgeMap == edgeMapDil;
	edgeMap.setTo(Scalar(0.0), edgeMask);


	if (!reverse)
	{
		for (int i = edgeMap.rows - 1; i >= 0; i--)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return i + 1;
	}
	else
	{
		for (int i = 0; i < edgeMap.rows; i++)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return edgeMap.rows - i;
	}

	return 0;
}

bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi)
{
	if (inImg.rows < 2 * (FRAME_MAX + 3) || inImg.cols < 2 * (FRAME_MAX + 3))
	{
		roi = Rect(0, 0, inImg.cols, inImg.rows);
		outImg = inImg;
		return false;
	}

	Mat imgGray;
	cvtColor(inImg, imgGray, CV_RGB2GRAY);

	int up, dn, lf, rt;
	
	up = findFrameMargin(imgGray.rowRange(0, FRAME_MAX), false);
	dn = findFrameMargin(imgGray.rowRange(imgGray.rows - FRAME_MAX, imgGray.rows), true);
	lf = findFrameMargin(imgGray.colRange(0, FRAME_MAX).t(), false);
	rt = findFrameMargin(imgGray.colRange(imgGray.cols - FRAME_MAX, imgGray.cols).t(), true);

	int margin = MAX(up, MAX(dn, MAX(lf, rt)));
	if ( margin == 0 )
	{
		roi = Rect(0, 0, imgGray.cols, imgGray.rows);
		outImg = inImg;
		return false;
	}

	int count = 0;
	count = up == 0 ? count : count + 1;
	count = dn == 0 ? count : count + 1;
	count = lf == 0 ? count : count + 1;
	count = rt == 0 ? count : count + 1;

	// cut four border region if at least 2 border frames are detected
	if (count > 1)
	{
		margin += 2;
		roi = Rect(margin, margin, inImg.cols - 2*margin, inImg.rows - 2*margin);
		outImg = Mat(inImg, roi);

		return true;
	}

	// otherwise, cut only one border
	up = up == 0 ? up : up + 2;
	dn = dn == 0 ? dn : dn + 2;
	lf = lf == 0 ? lf : lf + 2;
	rt = rt == 0 ? rt : rt + 2;

	
	roi = Rect(lf, up, inImg.cols - lf - rt, inImg.rows - up - dn);
	outImg = Mat(inImg, roi);

	return true;
}

Mat doWork(const Mat& src,bool use_lab, bool remove_border, bool use_geodesic )
{
	Mat Src = src;
	Mat src_small;
	float w = (float)src.cols, h = (float)src.rows;
	float maxD = max(w,h);
	resize(src,src_small,Size((int)(MAX_IMG_DIM*w/maxD),(int)(MAX_IMG_DIM*h/maxD)),0.0,0.0,INTER_AREA);// standard: width: 300 pixel
	Mat srcRoi;
	Rect roi;
	// detect and remove the artifical frame of the image
        if (remove_border)
		removeFrame(src_small, srcRoi, roi);
	else 
	{
		srcRoi = src_small;
		roi = Rect(0, 0, src_small.cols, src_small.rows);
	}


	if (use_lab)        
		cvtColor(srcRoi, srcRoi, CV_RGB2Lab);
	/* Computing saliency */
	MBS mbs(srcRoi);
	mbs.computeSaliency(use_geodesic);
	Mat resultRoiB = mbs.getSaliencyMap();

	int thres = getThreshold(resultRoiB, (int)(MAX_IMG_DIM*w / maxD), (int)(MAX_IMG_DIM*h / maxD));
	cv::Mat binaryMap = cv::Mat((int)(MAX_IMG_DIM*h / maxD), (int)(MAX_IMG_DIM*w / maxD), CV_8UC1);
	for (int i = 0; i < (int)(MAX_IMG_DIM*h / maxD)*(int)(MAX_IMG_DIM*w / maxD); i++)
	{
		if (resultRoiB.data[i] >= thres)
			binaryMap.data[i] = 255;
		else
			binaryMap.data[i] = 0;
	}
	Mat element = getStructuringElement(MORPH_CROSS, Size(4, 4));
	cv::erode(binaryMap, binaryMap, element);
	cv::dilate(binaryMap, binaryMap, element);
	Rect salrectRoi = getrectangular(binaryMap);
	float ratiox = int(w) / (MAX_IMG_DIM*w / maxD);
	float ratioy = int(h) / (MAX_IMG_DIM*h / maxD);

	salrectRoi.x = salrectRoi.x * ratiox + 0.5;
	salrectRoi.y = salrectRoi.y * ratioy + 0.5;
	salrectRoi.width = salrectRoi.width * ratiox + 0.5;
	salrectRoi.height = salrectRoi.height * ratioy + 0.5;

	/*cv::rectangle(Src, salrectRoi, CV_RGB(255, 0, 0), 2);
	cv::imshow("roi", Src);
	cv::waitKey(0);*/


	return binaryMap;
}

int main()
{
	string str;
	Mat src, dst;
	printf("plz input a image\n");
	cin >> str;
	src = imread(str);
	int timestart = clock();
	dst = doWork(src, true, true, false);
	int timeend = clock();
	int timecost = timeend - timestart;
	cout << timecost << endl;
	return 0;
}