#include <string>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/face.hpp>
#include <math.h>

#include <vector>
//#include <jni.h>
#include "SpecularHighlightRemoval.h"

#ifndef OPENCVEXTENDED_FACELANDMARKS_H
#define OPENCVEXTENDED_FACELANDMARKS_H

#define APPNAME "MainActivity"
#define LOG(TAG) __android_log_print(ANDROID_LOG_INFO, APPNAME, TAG);
#define SLOG(TAG) __android_log_print(ANDROID_LOG_INFO, APPNAME, "%s", TAG);

using namespace std;
using namespace cv;
using namespace cv::face;


enum landmarkPositions { JAWLINE, MOUTH, LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, NOSE };

class FaceLandmarks {
private:
    int getBiggest(vector<Rect> _selection);

    CascadeClassifier cascadeFace;
    Ptr<Facemark> facemark;
    vector<Point3f> modelPoints;
    vector<Point2f> landmarkSmallSet;

    

    Mat hue, hist, backProj;
    int hsize = 8;

    int hmin = 95, hmax = 256;
    int smin = 40, smax = 256;
    int vmin = 20, vmax = 256;

	//Wrinkle values
	const int MAX_ORIENTATION = 3;
	vector<Mat> gaborBank;
	double xmin[4], xmax[4];
	float orientations[3] = {
		-CV_PI / 2,
		CV_PI / 5,
		CV_PI / 2
	};

public:
    vector<vector<Point2f>> landmarks;
    Mat mask, image, gray, hsv, YCbCr, wrinkleMask, contour;
    bool faceLocated = 0;
    int tracking = 0;
	
	Rect face;
    
	FaceLandmarks();
    ~FaceLandmarks();

    void initialize(Mat _image, Mat _mask, Rect _face);
    void initialize(Mat _image, Mat _mask);

    //Where Facemark and Haar cascade get given there files to load
    void InitializeFacemark(string _modelAddress);
    bool InitializeHaar(string _haarAddress);

    //Finds a face using Haar Cascade and stores it in the face variable
    bool haarCascade();

    //Skin extraction techniques
    void scaleEllipse(Mat _gray);
    void colourSegmentation();
    Mat getSkinColour();
    void histogramProjection(Mat _hsv);
    bool lightingQuality();

    //Landmarking techniques
    bool findLandmarks();
    void placeLandmarks();
    void fillLandmarks(landmarkPositions l, Scalar colour);
    void headPositionEstimation();
    void headPositionInitialization();

    Mat getImage();
    void faceMask();
    Mat getMask();

};

#endif //OPENCVEXTENDED_FACELANDMARKS_H
