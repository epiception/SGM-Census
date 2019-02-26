#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "progressbar/ProgressBar.h"

using namespace std;
using namespace cv;

#define BLUR_RADIUS 3
#define PATHS_PER_SCAN 8
#define SMALL_PENALTY 3
#define LARGE_PENALTY 20
#define DEBUG false
#define window_height 9
#define window_width 9

struct path {
    short rowDiff;
    short colDiff;
    short index;
};


int image_height;
int image_width;


struct limits
{
    int start_pt_x; int start_pt_y;
    int end_pt_x; int end_pt_y;
    int direction_x; int direction_y;
};

vector<limits> paths;

void init_paths(int image_height, int image_width)
{
    //8 paths from center pixel based on change in X and Y coordinates
    for(int i =0;i < PATHS_PER_SCAN; i++)
    {
        paths.push_back(limits());
    }
    for(int i =0 ; i< PATHS_PER_SCAN; i++)
    {
        switch(i)
        {
            case 1:
            paths[i].direction_x = 0;
            paths[i].direction_y = 1;
            paths[i].start_pt_y = window_width/2;
            paths[i].end_pt_y = image_width - window_width/2;
            paths[i].start_pt_x = window_height/2;
            paths[i].end_pt_x = image_height - window_height/2;
            break;

            case 3:
            paths[i].direction_x = 0;
            paths[i].direction_y = -1;
            paths[i].start_pt_y = image_width - window_width/2;
            paths[i].end_pt_y = window_width/2;
            paths[i].start_pt_x = window_height/2;
            paths[i].end_pt_x = image_height - window_height/2;
            break;

            case 5:
            paths[i].direction_x = 1;
            paths[i].direction_y = -1;
            paths[i].start_pt_y = image_width - window_width/2;
            paths[i].end_pt_y = window_width/2;
            paths[i].start_pt_x = window_height/2;
            paths[i].end_pt_x = image_height - window_height/2;
            break;

            case 7:
            paths[i].direction_x = -1;
            paths[i].direction_y = -1;
            paths[i].start_pt_y = image_width - window_width/2;
            paths[i].end_pt_y = window_width/2;
            paths[i].start_pt_x = image_height - window_height/2;
            paths[i].end_pt_x = window_height/2;
            break;


            case 0:
            paths[i].direction_x = 1;
            paths[i].direction_y = 0;
            paths[i].start_pt_y = window_width/2;
            paths[i].end_pt_y = image_width - window_width/2;
            paths[i].start_pt_x = window_height/2;
            paths[i].end_pt_x = image_height - window_height/2;
            break;

            case 2:
            paths[i].direction_x = -1;
            paths[i].direction_y = 0;
            paths[i].start_pt_y = window_width/2;
            paths[i].end_pt_y = image_width - window_width/2;
            paths[i].start_pt_x = image_height - window_height/2;
            paths[i].end_pt_x = window_height/2;
            break;

            case 4:
            paths[i].direction_x = 1;
            paths[i].direction_y = 1;
            paths[i].start_pt_y = window_width/2;
            paths[i].end_pt_y = image_width - window_width/2;
            paths[i].start_pt_x = window_height/2;
            paths[i].end_pt_x = image_height - window_height/2;
            break;

            case 6:
            paths[i].direction_x = -1;
            paths[i].direction_y = 1;
            paths[i].start_pt_y = window_width/2;
            paths[i].end_pt_y = image_width - window_width/2;
            paths[i].start_pt_x = image_height - window_height/2;
            paths[i].end_pt_x = window_height/2;
            break;

            default:
            cout << "More paths or this is not possible" <<endl;
            break;

        }
    }
}

void calculateCostHamming(cv::Mat &firstImage, cv::Mat &secondImage, int disparityRange, unsigned long ***C, unsigned long ***S)
{
    unsigned long census_left = 0;
    unsigned long census_right = 0;
    unsigned int bit = 0;

    int bit_field=window_width*window_height-1;
    int i,j,x,y;
    int d=0;
    int shiftCount = 0;
    const int image_height=(int)firstImage.rows;
    const int image_width=(int)firstImage.cols;


    cout<<"size - ht: "<<image_height<<" wdt: "<<image_width<<endl;
    init_paths(image_height, image_width);

    Size imgSize = firstImage.size();
    Mat imgTemp_left = Mat::zeros(imgSize, CV_8U);
    Mat imgTemp_right = Mat::zeros(imgSize, CV_8U);
    Mat disparityMapstage1 = Mat(Size(firstImage.cols, firstImage.rows), CV_8UC1, Scalar::all(0));


    long census_vleft[image_height][image_width];
    long census_vright[image_height][image_width];

    cout<<"\ndisparity range is: "<< disparityRange<<endl;
    cout << "\nApplying Census Transform" <<endl;
    for (x = window_height/2; x < image_height - window_height/2; x++)
    {
        for(y = window_width/2; y < image_width - window_width/2; y++)
        {
            census_left = 0;
            shiftCount = 0;
            int bit_counter=0;
            int census_array_left[bit_field];
            for (i = x - window_height/2; i <= x + window_height/2; i++)
            {
                for (j = y - window_width/2; j <= y + window_width/2; j++)
                {

                    if( shiftCount != window_width*window_height/2 )//skip the center pixel
                    {
                        census_left <<= 1;
                        if( firstImage.at<uchar>(i,j) < firstImage.at<uchar>(x,y) )//compare pixel values in the neighborhood
                        bit = 1;
                        else
                        bit = 0;
                        census_left = census_left | bit;
                        census_array_left[bit_counter]=bit;bit_counter++;
                    }
                    shiftCount ++;
                }
            }

            imgTemp_left.ptr<uchar>(x)[y] = (short)census_left;
            census_vleft[x][y]=census_left;



            census_right = 0;
            shiftCount = 0;
            bit_counter=0;
            int census_array_right[bit_field];
            for (i = x - window_height/2; i <= x + window_height/2; i++)
            {
                for (j = y - window_width/2; j <= y + window_width/2; j++)
                {
                    if( shiftCount != window_width*window_height/2 )//skip the center pixel
                    {
                        census_right <<= 1;
                        if( secondImage.at<uchar>(i,j) < secondImage.at<uchar>(x,y) )//compare pixel values in the neighborhood
                        bit = 1;
                        else
                        bit = 0;
                        census_right = census_right | bit;
                        census_array_right[bit_counter]=bit;bit_counter++;
                    }
                    shiftCount ++;

                }
            }

            imgTemp_right.ptr<uchar>(x)[y] = (short)census_right;
            census_vright[x][y]=census_right;
        }

    }
    imwrite("Census_transform_output_left.png",imgTemp_left);
    imwrite("Census_transform_output_right.png",imgTemp_right);

    cout <<"\nFinding Hamming Distance" <<endl;
    for(x = window_height/2; x < image_height - window_height/2; x++)
    {
        for(y = window_width/2; y < image_width - window_width/2; y++)
        {
            for(int d=0;d<disparityRange;d++)
            {
                int census_left = 0;
                int  census_right = 0;
                shiftCount = 0;
                int bit_counter=0;
                census_left = census_vleft[x][y];
                if (y+d<image_width - window_width/2)
                    census_right= census_vright[x][y+d];
                else census_right= census_vright[x][y-disparityRange+d];
                long answer=(long)(census_left^census_right); //Hamming Distance
                short dist=0;
                while(answer)
                {
                    ++dist;
                    answer&=answer-1;
                }
                C[x][y][d]=dist;
            }
        }
    }

    for (int row = 0; row < firstImage.rows; ++row)
    {
        for (int col = 0; col < firstImage.cols; ++col)
        {
            unsigned long smallest_cost=C[row][col][0];
            long smallest_disparity=0;
            for(d=disparityRange-1;d>=0;d--)
            {
                if(C[row][col][d]<smallest_cost)
                {
                    smallest_cost=C[row][col][d];
                    smallest_disparity=d;
                }
            }

            disparityMapstage1.at<uchar>(row, col) = smallest_disparity*255.0/disparityRange; //Least cost Disparity
        }
    }

    imwrite("disparityMap_stage_1.png", disparityMapstage1);

}

void disprange_aggregation(int disparityRange,unsigned long ***C, unsigned int ****A, long unsigned last_aggregated_k, int direction_x, int direction_y, int curx, int cury, int current_path)
{
    long unsigned last_aggregated_i=C[curx][cury][0];

    for(int d=0;d<disparityRange;d++)
    {
        long unsigned term_1=A[current_path][curx-direction_x][cury-direction_y][0];
        long unsigned term_2=term_1;
        if(cury == window_width/2 || cury == image_width - window_width/2 || curx == window_height/2 || curx == image_height - window_height/2)
        {
            A[current_path][curx][cury][d]=C[curx][cury][d];
        }
        else
        {
            term_1=A[current_path][curx - direction_x][cury-direction_y][d];
            int limit_1,limit_2;
            if(d==0)
                term_2=A[current_path][curx - direction_x][cury - direction_y][d+1]+SMALL_PENALTY;

            else if(d==disparityRange-1)
                term_2=A[current_path][curx - direction_x][cury-direction_y][d-1]+SMALL_PENALTY;
            else
               term_2=min(A[current_path][curx - direction_x][cury-direction_y][d-1]+SMALL_PENALTY,
                          A[current_path][curx - direction_x][cury-direction_y][d+1]+SMALL_PENALTY);
            for(int pdisp=0;pdisp<disparityRange;pdisp++)
            {

                if((A[current_path][curx][cury-direction_y][pdisp]+LARGE_PENALTY)<term_1)
                    term_1=A[current_path][curx- direction_x][cury-direction_y][pdisp]+LARGE_PENALTY;
            }
            A[current_path][curx][cury][d]=C[curx][cury][d]+min(term_1,term_2)-last_aggregated_k;
        }
        if(A[current_path][curx][cury][d]<last_aggregated_i)
            last_aggregated_i=A[current_path][curx][cury][d];

    }
    last_aggregated_k=last_aggregated_i;
}


void aggregation(cv::Mat &firstImage, cv::Mat &secondImage, int disparityRange, unsigned long ***C, unsigned long ***S, unsigned int ****A)
{
    //Even and Odd paths based on change in X and Y coordinates
    ProgressBar bar(0);

    for(int ch_path = 0; ch_path < PATHS_PER_SCAN; ++ch_path)

    {
        long unsigned last_aggregated_k = 0;

        if(ch_path %2 !=0)
        {
            int dirx = paths[ch_path].direction_x;
            int diry = paths[ch_path].direction_y;
            int next_dim = 0;
            cout << "\n PATH: " << ch_path << endl;
            if(dirx == 0)
                next_dim = 1;
            else
                next_dim = dirx;
            bar.SetNIter(abs(paths[ch_path].start_pt_x - paths[ch_path].end_pt_x));
            bar.SetStyle('>');
            bar.Reset();
            for(int x=paths[ch_path].start_pt_x; x!=paths[ch_path].end_pt_x ;x+=next_dim)
            {
                bar.Update();
                for(int y=paths[ch_path].start_pt_y;( y!=paths[ch_path].end_pt_y);y+=diry)
                {
                    disprange_aggregation(disparityRange,C, A, last_aggregated_k, dirx, diry, x, y, ch_path);
                }
                std::this_thread::sleep_for( std::chrono::microseconds(300) );
            }
        }


        else if(ch_path%2 == 0)
        {
            int dirx = paths[ch_path].direction_x;
            int diry = paths[ch_path].direction_y;
            int next_dim = 0;
            cout << "\n PATH: " << ch_path << endl;
            if(diry == 0)
                next_dim = 1;
            else
                next_dim = diry;
            bar.SetNIter(abs(paths[ch_path].start_pt_y - paths[ch_path].end_pt_y));
            bar.SetStyle('>');
            bar.Reset();
            for(int y=paths[ch_path].start_pt_y; y!=paths[ch_path].end_pt_y ;y+=next_dim)
            {
                bar.Update();
                for(int x=paths[ch_path].start_pt_x;( x!=paths[ch_path].end_pt_x);x+=dirx)
                {
                    disprange_aggregation(disparityRange,C, A, last_aggregated_k, dirx, diry, x, y, ch_path);
                }
                std::this_thread::sleep_for( std::chrono::microseconds(300) );
            }

        }
    }

    cout << "\nAll paths covered" << endl;

    cout << "\nFinding summation term" << endl;

    for (int row = 0; row < firstImage.rows; ++row)
    {
        for (int col = 0; col < firstImage.cols; ++col)
        {
            for(int d = 0; d<disparityRange; d++)
            {
                for(int path = 0; path < PATHS_PER_SCAN; path ++)
                    S[row][col][d] += A[path][row][col][d]; //Aggregation
            }
        }
    }

}

void computeDisparity(int disparityRange, int rows, int cols, unsigned long ***S, char* out_file_name)
{
    Mat disparityMapstage2 = Mat(Size(cols, rows), CV_8UC1, Scalar::all(0));
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            unsigned long smallest_cost=S[row][col][0];
            int smallest_disparity=0;
            for(int d=disparityRange-1;d>=0;d--)
            {

                if(S[row][col][d]<smallest_cost)
                {
                    smallest_cost=S[row][col][d];
                    smallest_disparity=d; //Least cost disparity after Aggregation

                }
            }

            disparityMapstage2.at<uchar>(row, col) = smallest_disparity*255.0/disparityRange;

        }
    }

    imwrite(out_file_name, disparityMapstage2);
    cout <<"\nFin." <<endl;
}

int main(int argc, char** argv) {

    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <right image> <left image> <output image file> <disparity range>" << endl;
        return -1;
    }

    char *firstFileName = argv[1];
    char *secondFileName = argv[2];
    char *outFileName = argv[3];

    cv::Mat firstImage;
    cv::Mat secondImage;
    firstImage = cv::imread(firstFileName, CV_LOAD_IMAGE_GRAYSCALE);
    secondImage = cv::imread(secondFileName, CV_LOAD_IMAGE_GRAYSCALE);

    if(!firstImage.data || !secondImage.data) {
        cerr <<  "Could not open or find one of the images!" << endl;
        return -1;
    }

    unsigned int disparityRange = atoi(argv[4]);

    unsigned long ***C; // pixel cost array W x H x D
    unsigned long ***S; // aggregated cost array W x H x D
    unsigned int ****A; // single path cost array path_nos x W x H x D


    clock_t begin = clock();

    cout << "\nAllocating space..." << endl;


    // allocate cost arrays
    C = new unsigned long**[firstImage.rows];
    S = new unsigned long**[firstImage.rows];
    for (int row = 0; row < firstImage.rows; ++row) {
        C[row] = new unsigned long*[firstImage.cols];
        S[row] = new unsigned long*[firstImage.cols];
        for (int col = 0; col < firstImage.cols; ++col) {
            C[row][col] = new unsigned long[disparityRange]();
            S[row][col] = new unsigned long[disparityRange]();
        }
    }


    A = new unsigned int ***[PATHS_PER_SCAN];
    for(int path = 0; path < PATHS_PER_SCAN; ++path) {
        A[path] = new unsigned int **[firstImage.rows];
        for (int row = 0; row < firstImage.rows; ++row) {
            A[path][row] = new unsigned int*[firstImage.cols];
            for (int col = 0; col < firstImage.cols; ++col) {
                A[path][row][col] = new unsigned int[disparityRange];
                for (unsigned int d = 0; d < disparityRange; ++d) {
                    A[path][row][col][d] = 0;
                }
            }
        }
    }

    //Initial Smoothing
    GaussianBlur( firstImage, firstImage, Size( BLUR_RADIUS, BLUR_RADIUS ), 0, 0 );
    GaussianBlur( secondImage, secondImage, Size( BLUR_RADIUS, BLUR_RADIUS ), 0, 0 );

    cout << "\nCalculating pixel cost for the image..." << endl;
    calculateCostHamming(firstImage, secondImage, disparityRange, C, S);

    cout << "\nAggregating Costs" << endl;
    aggregation(firstImage, secondImage, disparityRange, C,S, A);

    cout << "\nComputing Disparity Map and Saving to Drive" << endl;
    computeDisparity(disparityRange, firstImage.rows, firstImage.cols,  S, outFileName);

    return 0;
}
