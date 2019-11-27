#include "gaborWrinkle.h"

WrinkleDetection::WrinkleDetection() {}
WrinkleDetection::~WrinkleDetection() {}

void WrinkleDetection::initialize(int _width, int _height, Rect _face) {

	face = _face;

	gray32 = Mat(_width, _height, CV_32F);
	kernel = Mat(_width, _height, CV_32F);
	gabor = Mat(_width, _height, CV_32F);
	gaborWeights = Mat(_width, _height, CV_32F);
	thresh = Mat(_width, _height, CV_8UC1);
	thresh2 = Mat(_width, _height, CV_8UC1);
	gray = Mat(_width, _height, CV_8UC1);
	lines = Mat(_width, _height, CV_8UC1);
	frame = Mat(_width, _height, CV_8UC1);
	contour = Mat(_width, _height, CV_8UC1);
	roughness = Mat(_width, _height, CV_8UC3);
	wrinkles = Mat(_width, _height, CV_8UC1);
	gaborFinal = Mat(_width, _height, CV_8U);
	score = Mat(_width, _height, CV_8UC3);


}

int WrinkleDetection::bresLineTracking(Point2f l1, Point2f l2) {
	int _x = l1.x;
	int _y = l1.y;

	int dx = l1.x - l2.x;
	int dy = l1.y - l2.y;

	int dLong = abs(dx);
	int dShort = abs(dy);

	int offLong = dx < 0 ? 1 : -1;
	int offShort = dy < 0 ? width : -width;

	if (dLong < dShort) {
		swap(dShort, dLong);
		swap(offShort, offLong);
	}

	int error = dLong / 2;
	int index = _y * width + _x;
	int offSet[] = { offLong, offLong + offShort };
	int absd[] = { dShort, dShort - dLong };

	//Tracks along the line and corrects errors such as going over an edge
	for (int k = 0; k <= dLong; k++) {
		//lines.ptr<unsigned char>()[index] = 255; //Draw after calculating length and curve of line
		const int tooBig = error >= dLong;
		index += offSet[tooBig];
		error += absd[tooBig];
	}
	return index;
}


void WrinkleDetection::run(Mat _image, Mat _mask) {

	wrinkles.setTo(Scalar(0, 0, 0));
	thresh.setTo(Scalar(0, 0, 0));
	thresh2.setTo(Scalar(0, 0, 0));
	score.setTo(Scalar(0, 0, 0));
	contour.setTo(Scalar(0, 0, 0));
	roughness.setTo(Scalar(0, 0, 0));
	gaborWeights.setTo(Scalar(0, 0, 0));
	lines.setTo(Scalar(0, 0, 0));

	frame = _image.clone();
	mask = _mask.clone();

	//Only gaussian blur an image that is too big
	if (face.area() > 100) {
		GaussianBlur(frame, frame, Size(7, 7), 1, 1);
	}
	cvtColor(frame, gray, COLOR_BGR2GRAY);

	gray.convertTo(gray32, CV_32F);

	//1. Apply Gabor filter bank to the image
	for (int i = 0; i < MAX_ORIENTATION; i++) {

		//Create a gabor kernel of 5x5 with gabor orientations
		kernel = getGaborKernel(Size(5, 5), 1, orientations[i], 150, 10);
		filter2D(gray32, gabor, CV_32F, kernel);

		//Apply 2x the colour, deepening any skin detail
		addWeighted(gaborWeights, 2.0, gabor, 2.0, 0, gaborWeights);
	}

	//Getting min and max of the basic filters and converts 32F to 8U
	minMaxIdx(gabor, xmin, xmax);
	gaborWeights.convertTo(gaborFinal, CV_8U, 255 / (xmax[0] - xmin[0]), -255 * xmin[0] / (xmax[0] - xmin[0]));

    int centre = 8;
    int step = 1;

    int startPixel = centre;
    int endPixel = (gaborFinal.rows) - centre;

	//Locate blobs of gradient changes
	parallel_for_(Range(startPixel, endPixel), [&](const Range range) -> void {

		int valueY = 0, valueX = 0;  //Values need to be inside parallel for loop

		//Locate any non-harsh changes in gradient in X direction
		for (int y = range.start; y < range.end; y += step) {
			for (int x = centre; x < gaborFinal.cols - centre; x++) {

				valueY *= gap;
				valueX *= gap;

				//Gets non-harsh gradients in a row, searches size - and +
				for (int i = -centre / 2; i < centre / 2; i++) {
					valueY += gaborFinal.at<unsigned char>(y, x) - gaborFinal.at<unsigned char>(y + i, x);
					valueY += gaborFinal.at<unsigned char>(y, x) - gaborFinal.at<unsigned char>(y - i, x);

					valueX += gaborFinal.at<unsigned char>(y, x) - gaborFinal.at<unsigned char>(y + i, x);
					valueX += gaborFinal.at<unsigned char>(y, x) - gaborFinal.at<unsigned char>(y - i, x);

					if (valueY >= t || valueX >= t) thresh.at<unsigned char>(y, x) += 30;	   //Threshold
					else if (-valueY >= t || -valueX >= t) thresh2.at<unsigned char>(y, x) += 30; //Inverse of threshold
				}

				//Threshold, removes colours below set amount
				if (thresh.at<unsigned char>(y, x) < 150) thresh.at<unsigned char>(y, x) = 0;
				if (thresh2.at<unsigned char>(y, x) < 150) thresh2.at<unsigned char>(y, x) = 0;

			}
		}
	}, 4);



    colour = 255;
	thresh2.copyTo(contour);

	//4. Remove areas such as blobs
	for (int l = 0; l < 2; l++) {
		if (l == 1) {
			thresh.copyTo(contour);
			colour = 150;
		}
		//Counts the area of the pixels
		for (int x = 0; x < contour.cols; x++) {
			for (int y = 0; y < contour.rows; y++) {

				if (contour.at<unsigned char>(y, x) > 0) {
					openQueue.push((Point2d(y, x)));
					contour.at<unsigned char>(y, x) = 0;
				}

				//Instead of counting each pixel find the size
				while (!openQueue.empty()) {

					//Find a way to store the data in a cleaner way
					for (int i = 0; i < 4; i++) {

						if (contour.at<unsigned char>(openQueue.front().x + searchLocations[i].x, openQueue.front().y + searchLocations[i].y) > 0) {
							openQueue.push(Point2d(openQueue.front().x + searchLocations[i].x, openQueue.front().y + searchLocations[i].y));
							contour.at<unsigned char>(openQueue.front().x + searchLocations[i].x, openQueue.front().y + searchLocations[i].y) = 0;

						}

					}

					//Save smallest Y and biggest Y value of each 
					closedQueue.push_back(openQueue.front());
					openQueue.pop();
				}

				if (closedQueue.size() > 200) { // Should change depending on resolution size

					//Method 1:  Place line in a mat on its own, so line tracking can be used here
					//Start with blob detection and locating a candidate line by locaitng the size

					for (int i = 0; i < closedQueue.size(); i++) {
						wrinkles.at<unsigned char>(closedQueue[i].x, closedQueue[i].y) = colour;
					}

				}

				closedQueue.clear();
			}
		}
	}



	//Starting point
	point2.x = 1;
	point2.y = 1;

	//width of image so that it cycles back if it goes over the edge
	width = lines.cols;

	//Start location i = total edges, appears facing left to right
	i = edges;

	for (int x = 0; x < wrinkles.cols; x++) {
		for (int y = 0; y < wrinkles.rows; y++) {

			if (wrinkles.at<unsigned char>(y, x) > 0) {
				openQueue.push(Point2d(y, x));
				colour = wrinkles.at<unsigned char>(y, x);
				point2.x = x;
				point2.y = y;
			}

			while (!openQueue.empty()) {
				//Theta ranges from 0.0 to 6.3

				//Places circles that are used to control trajectroy
				for (float theta = 0; theta < CV_2PI; theta += circleStep) {
					point1.x = point2.x + mainRadius * cos(theta);
					point1.y = point2.y - mainRadius * sin(theta);
					//lines.at<unsigned char>(point1.y, point1.x) = colour;
				}

				//Creates a circle around point 1 at the angle of i
				point1.x = point2.x + mainRadius * cos(circleStep * i);
				point1.y = point2.y - mainRadius * sin(circleStep * i);

				//Point 2 is previous point 1 is new point
				//if the new selected pixel is white then add it to the openQueue for next loop
				if (space < 5) {

					if (wrinkles.ptr<unsigned char>()[bresLineTracking(point2, point1)] != colour) space++;
					else space = 0;

					openQueue.push(Point2f(point1.y, point1.x));

					//Get the x and y of this location
					//dlong holds the final value of each line, turn 90 degrees from that point
					Point2d direction[2];
					int length[2];

					int rotation = (circleStep)* edges / 4;

					//Calculates 90 degree turn from current trajectory facing up
					direction[0].x = point1.x + outRadius * cos(circleStep*i + rotation);
					direction[0].y = point1.y - outRadius * sin(circleStep*i + rotation);

					//Calculates 90 degree turn from current trajectory facing down
					direction[1].x = point1.x + outRadius * cos(circleStep*i - rotation);
					direction[1].y = point1.y - outRadius * sin(circleStep*i - rotation);

					//Implementation of Bres line tracking
					//Cycles through points facing up and down and changes trajectory depending on line
					for (int l = 0; l < 2; l++) {
						_x = point1.x;
						_y = point1.y;

						dx = point1.x - direction[l].x;
						dy = point1.y - direction[l].y;

						dLong = abs(dx);
						dShort = abs(dy);

						offLong = dx < 0 ? 1 : -1;
						offShort = dy < 0 ? width : -width;

						if (dLong < dShort) {
							swap(dShort, dLong);
							swap(offShort, offLong);
						}

						index = _y * width + _x;

						error = dLong / 2;
						offSet[0] = offLong;
						offSet[1] = offLong + offShort;

						absd[0] = dShort;
						absd[1] = dShort - dLong;

						lineCount = 0;

						//Tracks along the line and corrects errors such as going over an edge
						for (int k = 0; k <= dLong; k++) {
							if (wrinkles.ptr<unsigned char>()[index] == colour) {
								lineCount++;
							}

							//lines.ptr<unsigned char>()[index] = 255;
							const int tooBig = error >= dLong;
							index += offSet[tooBig];
							error += absd[tooBig];
						}

						length[l] = lineCount;
					}

					//Moves the trajectory up and down
					i += (length[0] - length[1]) * 1.2;

				}

				//Changes the previous centre location of the new circle
				point2.x = point1.x;
				point2.y = point1.y;
				linePoints.push_back(openQueue.front());

				openQueue.pop();
			}
			space = 0;
			i = edges;

			//Once finished tracking particular line remove all connected areas of the same colour
			//Track from the defined points
			if (linePoints.size() > 50) {
				for (int l = 0; l < linePoints.size(); l++) {
					//bresLineTracking(linePoints[l - 1], linePoints[l]);
					lines.at<unsigned char>(linePoints[l].x, linePoints[l].y) = colour;
				}
			}

			//Finds all connected areaqs associated with the line points and removes the discovered line from the image
			//Slow, remove in the future
			while (!linePoints.empty()) {

				for (int j = 0; j < 4; j++) {

					if (wrinkles.at<unsigned char>(linePoints.front().x + searchLocations[j].x, linePoints.front().y + searchLocations[j].y) == colour) {
						linePoints.push_back(Point2d(linePoints.front().x + searchLocations[j].x, linePoints.front().y + searchLocations[j].y));
						wrinkles.at<unsigned char>(linePoints.front().x + searchLocations[j].x, linePoints.front().y + searchLocations[j].y) = 0;

					}
				}

				linePoints.erase(linePoints.begin());
				//linePoints.pop_back();
			}


		}
	}


    linePoints.clear();
}

Mat WrinkleDetection::getWrinkles() {
	return wrinkles;
}

Mat WrinkleDetection::getRoughness() {

	return roughness;
}

Mat WrinkleDetection::getGabor() {
	return gaborFinal;
}

Mat WrinkleDetection::getThresh() {
	return thresh;
}

Mat WrinkleDetection::getLines() {
	return lines;
}