// Main.cpp
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "Main.h"
//
// #define CAP_WIDTH   320
// #define CAP_HEIGHT  240

void OpenCamera();

///////////////////////////////////////////////////////////////////////////////////////////////////
/** @function main */
//int main( int argc, char** argv ) {     //if you want to load image by typing the filename

int main(void) {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();           // attempt KNN training

    if (blnKNNTrainingSuccessful == false) {                            // if KNN training was not successful
                                                                        // show error message
        std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
        return(0);                                                      // and exit program
    }

    OpenCamera();

    cv::Mat imgOriginalScene;           // input image

    imgOriginalScene = cv::imread("camera.png");         // open image

    /// Load an image
    //imgOriginalScene = cv::imread( argv[1] );

    if (imgOriginalScene.empty()) {                             // if unable to open image
        std::cout << "error: image not read from file\n\n";     // show error message on command line
        // _getch();                                               // may have to modify this line if not using Windows
        return(0);                                              // and exit program
    }

    cv::Mat imgInvert = imgOriginalScene;

    cv::bitwise_not(imgOriginalScene, imgInvert);

    std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgInvert);          // detect plates

    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               // detect chars in plates

    cv::bitwise_not(imgInvert, imgOriginalScene);
    cv::namedWindow("imgOriginalScene",CV_WINDOW_NORMAL);
    cv::imshow("imgOriginalScene", imgOriginalScene);           // show scene image, imgOriginalScene

    if (vectorOfPossiblePlates.empty()) {                                               // if no plates were found
        std::cout << std::endl << "no license plates were detected" << std::endl;       // inform user no plates were found
    }
    else {                                                                            // else
                                                                                      // if we get in here vector of possible plates has at leat one plate

                                                                                      // sort the vector of possible plates in DESCENDING order (most number of chars to least number of chars)
        std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);

        // suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        PossiblePlate licPlate = vectorOfPossiblePlates.front();

        cv::namedWindow("imgPlate",CV_WINDOW_NORMAL);
        cv::imshow("imgPlate", licPlate.imgPlate);            // show crop of plate and threshold of plate
        cv::namedWindow("imgThresh",CV_WINDOW_NORMAL);
        cv::imshow("imgThresh", licPlate.imgThresh);

        if (licPlate.strChars.length() == 0) {                                                      // if no chars were found in the plate
            std::cout << std::endl << "no characters were detected" << std::endl << std::endl;      // show message
            return(0);                                                                              // and exit program
        }

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate);                // draw red rectangle around plate, imgOriginalScene

        std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;     // write license plate text to std out
        std::cout << std::endl << "-----------------------------------------" << std::endl;

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);              // write license plate text on the image, imgOriginalScene

        cv::namedWindow("imgOriginalScene",CV_WINDOW_NORMAL);
        cv::imshow("imgOriginalScene", imgOriginalScene);                       // re-show scene image

        cv::imwrite("imgOriginalScene.png", imgOriginalScene);                  // write image out to file
    }

    cv::waitKey(0);                 // hold windows open until user presses a key

    return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {    //imgOriginalScene
    cv::Point2f p2fRectPoints[4];

    licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);            // get 4 vertices of rotated rect

    for (int i = 0; i < 4; i++) {                                       // draw 4 red lines
        cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 3);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point ptCenterOfTextArea;                   // this will be the center of the area the text will be written to
    cv::Point ptLowerLeftTextOrigin;                // this will be the bottom left of the area that the text will be written to

    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;                              // choose a plain jane font
    double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            // base font scale on height of plate area
    int intFontThickness = (int)std::round(dblFontScale * 1.5);             // base font thickness on font scale
    int intBaseline = 0;

    cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);      // call getTextSize

    ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         // the horizontal location of the text area is the same as the plate

    if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      // if the license plate is in the upper 3/4 of the image
                                                                                            // write the chars in below the plate
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    else {                                                                                // else if the license plate is in the lower 1/4 of the image
                                                                                          // write the chars in above the plate
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }

    ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));           // calculate the lower left origin of the text area
    ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));          // based on the text area center, width, and height

                                                                                            // write the text on the image
    cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}

void OpenCamera()
{
    // Start capturing on camera 1
    cv::VideoCapture cap(1);
    if(!cap.isOpened())
        printf("ERROR! Camera not opened!\n");

    // This matrix will store the edges of the captured frame
    cv::Mat frame;

    while(1)
    {

    // Acquire the frame from cap into frame
    cap >> frame;

    // Display the frame
    cv::imshow("camera", frame);

    // Terminate by pressing a key
     if(cv::waitKey(30) == 27){
         cap.release();
         break;
     }

    }

    // Write image
    cv::imwrite("camera.png", frame);
    cap.release();

}

/*
algoritma buat autofokus
pindahin ke Rp
laporan kp
*/
