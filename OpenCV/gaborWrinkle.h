#pragma once


#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace cv;
using namespace std;

class WrinkleDetection
{
public:

	WrinkleDetection();
	~WrinkleDetection();

	void initialize(int width, int height, Rect _face);
	void run(Mat image, Mat mask);
	int bresLineTracking(Point2f l1, Point2f l2);

	Mat frame, hist, gray, gray32, thresh, thresh2, contour, wrinkles, roughness, score, gabor, gaborWeights, gaborFinal, kernel, mask, lines;
	Rect face;

	int t = 130;
	int width;

	const int MAX_ORIENTATION = 3;

	std::vector<Mat> gaborBank;
	double xmin[4], xmax[4];

	//2. Create and apply a gabor filter bank
	float orientations[3] = {
		-CV_PI / 2,
		CV_PI / 5,
		CV_PI / 2
	};
	
	Mat getWrinkles();
	Mat getRoughness();
	Mat getGabor();
	Mat getThresh();
	Mat getLines();

private:



	//filtering
	int blobSize = 50;
	int val = 0;

	Point2d searchLocations[8] = {
			{1, 0}, {0, 1},
			{-1, 0}, {0, -1},
			{-1, 1}, {1, -1},
			{-1, -1}, {1, 1}
	};

	int centre = 8;
	int step = 1;

	int startPixel = centre;
	int endPixel = (gaborFinal.rows) - centre;

	float gap = 0.1;  //Use carefully, this will increase the score and make it harder to remove noise

	//Removing blobs
	queue<Point2d> openQueue;
	vector<Point2d> closedQueue;
	int i = 1, j = 1, k = 0, count = 0, colour = 0;

	float add = 0;


	//Line tracking
	Point2f point1;
	Point2f point2;

	//Amount of edges around the defined circles
	int edges = 64;
	float circleStep = CV_2PI / edges;

	//Increase to skip gaps when tracking accross line
	//Greatly decreases speed for bigger images
	int mainRadius = 2;

	int outRadius = 5;

	Size shape;

	int _x, _y, dx, dy, dLong, dShort,
	offLong, offShort, error, index, offSet[2], absd[2], lineCount;
	int space = 0;

	vector<Point2d> linePoints;
};

