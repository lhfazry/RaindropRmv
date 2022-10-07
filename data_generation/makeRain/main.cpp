#include <iostream>
#include <random>
#include <glob.h>
#include <regex>
#include <boost/format.hpp>

#include "rain.h"

using namespace std;
using boost::format;
using boost::str;


void getFiles(const string &pattern, vector<string> &filePath);
int new_width = 128;
int new_height = 128;

int main(int argc, char** argv) {
    map<string, double> params;
    vector<string> imgPath;
    getFiles(str(format("%1%/*/*/*.png")%argv[1]), imgPath);

    //getFiles("repo/dataset/cityscapes/leftImage/"
    //        "train/*/*.png", imgPath);
 
    // getFiles("/media/ros/Workshop/ws/Datasets/cityscapes/leftImage/"
    //          "val/*/*.png", imgPath);

    //getFiles("~/repo/dataset/cityscapes/leftImage/"
    //         "test/*/*.png", imgPath);
    //getFiles("~/repo/dataset/cityscapes/leftImage/test/mainz/*.png", imgPath);

    // number of images each original image produces
    int totalIndex = imgPath.size(), numsPerImg{5};
    cout << totalIndex << endl;

    // some random number generator
    mt19937 rng;
    rng.seed(random_device()());
    uniform_int_distribution<int> random_M(100, 500);
    uniform_int_distribution<int> random_B(4000, 8000);
    uniform_int_distribution<int> random_psi(30, 45);
    uniform_int_distribution<int> random_dia(3, 20);   // blur kernel size

    string savePath{argv[2]};

    for(int index{0}; index < totalIndex; ++index) {
        if(index % 10 == 0) {
            cout << "Processing: " << setprecision(2) << index << " / " << totalIndex << " (" << float(index) / totalIndex << ")" << endl;
        }
        unsigned count{0};
        params["M"] = random_M(rng);
        params["B"] = random_B(rng);
    
        // params["M"] = 100;
        // params["B"] = 8000;
        params["psi"] = random_psi(rng);
        Rain rain(params, imgPath[index]);

        cv::Mat img;

        cout << str(format("resize rain.image, original: h: %1%, w: %2%")%rain.image.rows%rain.image.cols) << endl;
        cv::resize(rain.image, img, cv::Size(new_width, new_height), cv::INTER_NEAREST);
        cv::imwrite(str(format("%1%/%2%_I.png")%savePath%index), img);
//        cv::imshow("test_show input", img);

        for(int i{0}; i < numsPerImg; i++) {

            cout << "render" << endl;
            rain.render();      
            cv::Mat rain_img;

            cout << str(format("resize rain.rain_image, original: h: %1%, w: %2%")%rain.rain_image.rows%rain.rain_image.cols) << endl;
            cv::resize(rain.rain_image, rain_img, cv::Size(new_width, new_height), cv::INTER_NEAREST);
            cv::imwrite(str(format("%1%/%2%_I.png")%savePath%index), img);
//            cv::imshow("test_show rain", rain_img);
            auto kernel = rain.get_kernel(random_dia(rng));
            rain.blur(kernel);

            cout << str(format("resize rain.mask, original: h: %1%, w: %2%")%rain.mask.rows%rain.mask.cols) << endl;
            cv::Mat mask, blur;
            cv::resize(rain.mask, mask, cv::Size(new_width, new_height), cv::INTER_NEAREST);

            cout << str(format("resize rain.blur_image, original: h: %1%, w: %2%")%rain.blur_image.rows%rain.blur_image.cols) << endl;
            cv::resize(rain.blur_image, blur, cv::Size(new_width, new_height), cv::INTER_NEAREST);

            cv::imwrite(str(format("%1%/%2%_%3%_M.png")%savePath%index%count), mask);
            cv::imwrite(str(format("%1%/%2%_%3%_B.png")%savePath%index%count), blur);
            
            cout << str(format("imgPath[%1%] = %2%")%index%imgPath[index]) << endl;
            std::string path_sem = std::regex_replace(imgPath[index], regex(R"(leftImg8bit)"), "gtFine");
            std::string path_sem_seg = std::regex_replace(path_sem, regex(R"(_gtFine)"), "_gtFine_labelIds");
            std::string path_ins_seg = std::regex_replace(path_sem, regex(R"(_gtFine)"), "_gtFine_instanceIds");
            std::string path_sem_seg_color = std::regex_replace(path_sem, regex(R"(_gtFine)"), "_gtFine_color");
            // std::cout << path_sem_seg << std::endl;
            // std::cout << path_sem_seg_color << std::endl;
            cout << "path_sem: " << path_sem << endl;
            cout << "path_sem_seg: " << path_sem_seg << endl;
            cout << "path_ins_seg: " << path_ins_seg << endl;
            cout << "path_sem_seg_color: " << path_sem_seg_color << endl;

            cv::Mat sem = cv::imread(path_sem_seg, -1);
            cv::Mat sem_save;

            cout << str(format("resize sem, original: h: %1%, w: %2%")%sem.rows%sem.cols) << endl;
            cv::resize(sem, sem_save, cv::Size(new_width, new_height), cv::INTER_NEAREST);
            cv::imwrite(str(format("%1%/%2%_%3%_S.png")%savePath%index%count), sem_save);
            sem = cv::imread(path_sem_seg_color);
            sem_save;

            cout << str(format("resize sem, original: h: %1%, w: %2%")%sem.rows%sem.cols) << endl;
            cv::resize(sem, sem_save, cv::Size(new_width, new_height), cv::INTER_NEAREST);
            cv::imwrite(str(format("%1%/%2%_%3%_S_color.png")%savePath%index%count), sem_save);
            sem = cv::imread(path_ins_seg, -1);
            sem_save;

            cout << str(format("resize sem, original: h: %1%, w: %2%")%sem.rows%sem.cols) << endl;
            cv::resize(sem, sem_save, cv::Size(new_width, new_height), cv::INTER_NEAREST);
            cv::imwrite(str(format("%1%/%2%_%3%_Ins.png")%savePath%index%count), sem_save);
//            cv::imshow("test show mask", mask);
//            cv::imshow("test show blur", blur);
//
//            cv::waitKey();

            //cv::imwrite(str(format("%1%/%2%_%3%_B.png")%savePath%index%count), rain.blur_image);
            //cv::imwrite(str(format("%1%/%2%_%3%_M.png")%savePath%index%count), rain.mask);
            ++count;
        }
    }
    return 0;
}

void getFiles(const string &pattern, vector<string> &filePath) {
    glob_t globBuf;

    glob(pattern.c_str(), GLOB_TILDE, NULL, &globBuf);
    
    for(unsigned i{0}; i < globBuf.gl_pathc; i++) {
        filePath.push_back(globBuf.gl_pathv[i]);
    }

    if(globBuf.gl_pathc > 0) {
        globfree(&globBuf);
    }
}

