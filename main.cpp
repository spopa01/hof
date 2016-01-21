//g++ main.cpp -o experiment -I/usr/local/Cellar/opencv/2.4.12/include/ -L/usr/local/Cellar/opencv/2.4.12/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_ml -lopencv_objdetect -lopencv_video -lboost_system -lboost_filesystem -std=c++11

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdlib>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

bool is_file(std::string& p){
    boost::filesystem::path path{p};
    if (boost::filesystem::exists(path) && boost::filesystem::is_regular_file(path))
        return true;
    return false;
}

bool is_directory(std::string& p){
    boost::filesystem::path path{p};
    if (boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
        return true;
    return false;
}

std::string check_file(std::string p){
    if (!is_file(p))
        throw std::invalid_argument( boost::str(boost::format("invalid file: %1%") % p) );
    return p;
}

std::string check_directory(std::string p){
    if (!is_directory(p))
        throw std::invalid_argument( boost::str(boost::format("invalid directory: %1%") % p) );
    return p;
}

int check_samples(int n){
    if(n > 32)
        throw std::invalid_argument("too many samples");
    return n;
}

const uint64_t dct_blk_sz = 8;
const uint64_t dct_blk_sz2 = 64;

struct selection_mask_t {
    int element[dct_blk_sz][dct_blk_sz];
    int bins{0};

    bool use_pos_neg{false};
    bool use_abs_val{false};

    void compute_bins() {
        std::set<int> bins;
        for (int i = 0; i < dct_blk_sz; ++i) {
            for (int j = 0; j < dct_blk_sz; ++j) {
                if (element[i][j] > 0)
                    bins.insert(element[i][j]);
            }
        }
        this->bins = bins.size();
    }

    void check_bins() {
        if (bins == 0)
            throw std::invalid_argument("invalid bin selection 0");

        for (int i = 0; i < dct_blk_sz; ++i) {
            for (int j = 0; j < dct_blk_sz; ++j) {
                if (element[i][j] < 0 || element[i][j] > bins)
                    throw std::invalid_argument("invalid bin selection X");
            }
        }
    }
};

std::istream& operator>>(std::istream& is, selection_mask_t& sm){
    for(int l=0; l<8; ++l){
        for(int c=0; c<8; ++c){
            is >> sm.element[l][c];
        }
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, selection_mask_t const& sm){
    for(int l=0; l<8; ++l){
        for(int c=0; c<8; ++c){
            os << sm.element[l][c] << "\t";
        }
        std::cout << std::endl;
    }
    return os;
}

const cv::Point train_window_stride{16, 16};
const cv::Point test_window_stride{3, 3};
const cv::Size window_size{64, 128};
const cv::Size window_size_in8x8{8, 16};

template<typename T>
static void print_block(cv::Mat const& block, cv::Rect win){
    for(int h = win.y; h < win.y + win.height; ++h){
        for(int w = win.x; w < win.x + win.width; ++w){
            std::cout << block.at<T>(h,w) << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
static void print_block(cv::Mat const& block){
    print_block<T>(block, cv::Rect(cv::Size(0,0), block.size()));
}

struct experiment_t {
    std::string dataset_path;
    std::string selection_mask_path;

    selection_mask_t dct_selection_mask;

    //number of sampled windows (per height and width) per negative image (train and test)
    cv::Size windows_per_negative_sample;
    int num_windows_per_negative_sample;

    std::vector<std::string> train_positive_img_names;
    std::vector<std::string> train_negative_img_names;
    std::vector<std::string> test_positive_img_names;
    std::vector<std::string> test_negative_img_names;

    static void load_names(std::string file, std::string from, std::string to, std::vector<std::string>& file_names){
        std::ifstream in_file(check_file(file));
        std::string file_name;
        while(std::getline(in_file, file_name)){
            if (file_name.empty()) continue;
            auto name = boost::replace_all_copy( boost::replace_all_copy( boost::replace_all_copy(file_name, from, to), "png", "bmp"), "jpg", "bmp" );
            file_names.push_back(check_file(name));
        }
    }

    std::vector<cv::Mat> train_positive_imgs;
    std::vector<cv::Mat> train_negative_imgs;
    std::vector<cv::Mat> test_positive_imgs;
    std::vector<cv::Mat> test_negative_imgs;

    cv::Mat train_features;
    cv::Mat train_labels;

    cv::Mat test_pos_features;
    cv::Mat test_neg_features;
    cv::Mat expected_pos_test_labels;
    cv::Mat expected_neg_test_labels;
    cv::Mat actual_pos_test_labels;
    cv::Mat actual_neg_test_labels;

    static void load_images(std::vector<std::string> const& file_names, std::vector<cv::Mat>& files){
        std::for_each(file_names.begin(), file_names.end(),
                      [&files](std::string const& file_name){
                          files.emplace_back(cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE));
                      });
    }

    static void explode_in_blocks(cv::Mat const& image, cv::Size block_size, std::vector<std::vector<cv::Mat>>& blocks){
        int num_blocks_w = image.size().width/block_size.width;
        int num_blocks_h = image.size().height/block_size.height;

        for (int h = 0; h < num_blocks_h; ++h) {
            blocks.emplace_back();
            for (int w = 0; w < num_blocks_w; ++w) {
                blocks[h].emplace_back(image, cv::Rect(cv::Point(w*block_size.width, h*block_size.height), block_size));
            }
        }
    }

    static void clone_blocks(std::vector<std::vector<cv::Mat>> const& blocks_in, std::vector<std::vector<cv::Mat>>& blocks_out){
        for (int h = 0; h < blocks_in.size(); ++h) {
            blocks_out.emplace_back();
            for (int w = 0; w < blocks_in[h].size(); ++w) {
                blocks_out[h].push_back(blocks_in[h][w].clone());
            }
        }
    }

    static void convert_blocks(std::vector<std::vector<cv::Mat>> const& blocks_in, std::vector<std::vector<cv::Mat>>& blocks_out, int type){
        for (int h = 0; h < blocks_in.size(); ++h) {
            blocks_out.emplace_back();
            for (int w = 0; w < blocks_in[h].size(); ++w) {
                blocks_out[h].emplace_back();
                blocks_in[h][w].convertTo(blocks_out[h][w], type);
            }
        }
    }

    static void apply_dct_to_blocks(std::vector<std::vector<cv::Mat>>& blocks){
        for (int h = 0; h < blocks.size(); ++h) {
            for (int w = 0; w < blocks[h].size(); ++w) {
                cv::dct(blocks[h][w].clone(), blocks[h][w]);
            }
        }
    }

    static void quantize_dct_blocks(std::vector<std::vector<cv::Mat>>& blocks){
        //assert(blocks type && size);
        static cv::Mat qblock =
        (cv::Mat_<int>(8, 8) << 16,  11,  10,  16,  24,   40,   51,   61,
                                12,  12,  14,  19,  26,   58,   60,   55,
                                14,  13,  16,  24,  40,   57,   69,   56,
                                14,  17,  22,  29,  51,   87,   80,   62,
                                18,  22,  37,  56,  68,   109,  103,  77,
                                24,  35,  55,  64,  81,   104,  113,  92,
                                49,  64,  78,  87,  103,  121,  120,  101,
                                72,  92,  95,  98,  112,  100,  103,  99);

        for (int h = 0; h < blocks.size(); ++h) {
            for (int w = 0; w < blocks[h].size(); ++w) {
                blocks[h][w] = blocks[h][w] / qblock;
            }
        }
    }

    //from x by y to (xy) by 1
    static void extract_block_feature(cv::Mat const& block, selection_mask_t const& dct_selection_mask, cv::Mat& feature_block){
        //assert elements of block as integers and the size of the block 8x8
        for(int h = 0; h < block.size().height; ++h){
            for(int w = 0; w < block.size().width; ++w){
                auto val = block.at<int>(h, w);

                if (dct_selection_mask.element[h][w] == 0)
                    continue;

                int idx = dct_selection_mask.use_pos_neg ?
                          2 * (dct_selection_mask.element[h][w]-1)+(val > 0 ? 1 : 0) :
                          (dct_selection_mask.element[h][w]-1);

                feature_block.at<int>(0, idx) += (!dct_selection_mask.use_abs_val ? val : std::abs(val));
            }
        }
    }

    //HOG cells equiv.
    static void extract_basic_block_features(std::vector<std::vector<cv::Mat>>& blocks, selection_mask_t const& dct_selection_mask, std::vector<std::vector<cv::Mat>>& basic_block_features){
        for (int h = 0; h < blocks.size(); ++h) {
            basic_block_features.emplace_back();
            for (int w = 0; w < blocks[h].size(); ++w) {
                basic_block_features[h].emplace_back(cv::Size(dct_selection_mask.bins * (dct_selection_mask.use_pos_neg ? 2 : 1), 1), CV_32SC1, cv::Scalar(0));
                extract_block_feature(blocks[h][w], dct_selection_mask, basic_block_features[h][w]);
            }
        }
    }

    //HOG blocks equiv.  2x2 features.
    static void build_advanced_block_features(std::vector<std::vector<cv::Mat>> const& basic_block_features, std::vector<std::vector<cv::Mat>>& advanced_block_features){
        auto feat_size = basic_block_features[0][0].size();
        //assert feat_size height == 1
        for (int h = 0; h < basic_block_features.size()-1; ++h) {
            advanced_block_features.emplace_back();
            for (int w = 0; w < basic_block_features[h].size()-1; ++w) {
                advanced_block_features[h].emplace_back(cv::Size(4 * feat_size.width, 1), CV_32FC1);

                basic_block_features[h][w].copyTo(cv::Mat(advanced_block_features[h][w], cv::Rect(cv::Point(0,0), cv::Size(feat_size.width, 1))));
                basic_block_features[h][w+1].copyTo(cv::Mat(advanced_block_features[h][w], cv::Rect(cv::Point(feat_size.width, 0), cv::Size(feat_size.width, 1))));
                basic_block_features[h+1][w].copyTo(cv::Mat(advanced_block_features[h][w], cv::Rect(cv::Point(2*feat_size.width, 0), cv::Size(feat_size.width, 1))));
                basic_block_features[h+1][w+1].copyTo(cv::Mat(advanced_block_features[h][w], cv::Rect(cv::Point(3*feat_size.width, 0), cv::Size(feat_size.width, 1))));

                //here we can experiment with more norms...
                cv::normalize(advanced_block_features[h][w], advanced_block_features[h][w], 1, 0, cv::NORM_INF);
            }
        }
    }

    static void build_final_feature(std::vector<std::vector<cv::Mat>> const& advanced_block_features, cv::Mat& features, int idx)
    {
        //advanced_block_features[0][0].size().height = 1
        int advanced_feature_size = advanced_block_features[0][0].size().area();
        for (int h = 0; h < advanced_block_features.size(); ++h) {
            for (int w = 0; w < advanced_block_features[h].size(); ++w) {
                auto line_offset = (int)((h * advanced_block_features[0].size() + w) * advanced_feature_size);
                cv::Mat hdr(features, cv::Rect(cv::Point(line_offset, idx), cv::Size(advanced_feature_size, 1)));
                advanced_block_features[h][w].copyTo(hdr);
            }
        }
    }

    static void extract_dct_feature(cv::Mat const& window, selection_mask_t const& dct_selection_mask, cv::Mat& features, int idx){
        //convert the image to a matrix of floats
        cv::Mat window_f; window.convertTo(window_f, CV_32FC1);

        //explode it into blocks of 8x8 in order to apply dct
        std::vector<std::vector<cv::Mat>> blocks_8x8_f;
        explode_in_blocks(window_f, cv::Size(8,8), blocks_8x8_f);

        //apply dct per blocks
        apply_dct_to_blocks(blocks_8x8_f);

        //convert the resul to integers and quantize the result
        std::vector<std::vector<cv::Mat>> blocks_8x8_i;
        convert_blocks(blocks_8x8_f, blocks_8x8_i, CV_32SC1);
        quantize_dct_blocks(blocks_8x8_i);

        //extract features per block using the selection matrix
        std::vector<std::vector<cv::Mat>> basic_block_features;
        extract_basic_block_features(blocks_8x8_i, dct_selection_mask, basic_block_features);

        //compute the advanced features
        std::vector<std::vector<cv::Mat>> advanced_block_features;
        build_advanced_block_features(basic_block_features, advanced_block_features);

        //add one more line to the features matrix
        build_final_feature(advanced_block_features, features, idx);
    }

    static void extract_middle_window(cv::Mat const& image, selection_mask_t const& dct_selection_mask, cv::Point const& window_stride, cv::Mat& features, int& idx){
        extract_dct_feature(cv::Mat(image, cv::Rect(window_stride, window_size)), dct_selection_mask, features, idx++);
        //cv::imshow("", windows.back()); cv::waitKey();
    }

    static void compute_offsets(std::vector<int>& offsets, int image_size, int window_size, int window_samples){
        bool overlapping = image_size < window_size *  window_samples;
        int total_windows_len = window_size *  window_samples;
        int diff = (!overlapping ? (image_size < total_windows_len) : (total_windows_len - image_size));

        int off = (!overlapping ? 1 : -1) * ((diff/(window_samples-1)) + (!overlapping ? -1 : 1));
        for (int i = 0; i < window_samples; ++i) {
            offsets.push_back(i * (window_size + off));
        }
    }

    static void extract_matrix_windows(cv::Mat const& image, selection_mask_t const& dct_selection_mask, cv::Size const& window_samples, cv::Mat& features, int& idx){
        std::vector<int> offsets_w;
        std::vector<int> offsets_h;

        compute_offsets(offsets_w, image.cols, window_size.width, window_samples.width);
        compute_offsets(offsets_h, image.rows, window_size.height, window_samples.height);

        for (auto off_h : offsets_h) {
            for (auto off_w : offsets_w) {
                extract_dct_feature(cv::Mat(image, cv::Rect(cv::Point(off_w, off_h), window_size)), dct_selection_mask, features, idx++);
                //cv::imshow("", windows.back()); cv::waitKey();
            }
        }
    }

    experiment_t(std::string dir_path, std::string mask_path, int neg_w_samples, int neg_h_samples, bool use_pos_neg, bool use_abs_val)
        : dataset_path{dir_path}, selection_mask_path{ mask_path }
    {
        windows_per_negative_sample = cv::Size{neg_w_samples, neg_h_samples};
        num_windows_per_negative_sample = neg_w_samples * neg_h_samples;

        dct_selection_mask.use_pos_neg = use_pos_neg;
        dct_selection_mask.use_abs_val = use_abs_val;
    }

    void load_data(){
        std::cout<<"start loading data"<<std::endl;

        std::ifstream selection_mask_file(selection_mask_path);
        selection_mask_file >> dct_selection_mask;
        dct_selection_mask.compute_bins();
        dct_selection_mask.check_bins();

        if(dataset_path.back() != '/') dataset_path.push_back('/');

        static std::string pos      = "pos.lst";
        static std::string neg      = "neg.lst";
        static std::string train    = "train";
        static std::string test     = "test";
        static std::string cst      = "_64x128_H96/";

        static std::string full_train   = dataset_path + train + cst;
        static std::string full_test    = dataset_path + test  + cst;

        static std::string full_train_pos   = full_train + pos;
        static std::string full_train_neg   = full_train + neg;
        static std::string full_test_pos    = full_test + pos;
        static std::string full_test_neg    = full_test + neg;

        load_names( full_train_pos,  train+"/",  full_train, train_positive_img_names   );
        load_names( full_train_neg,  train+"/",  full_train, train_negative_img_names   );
        load_names( full_test_pos,   test+"/",   full_test,  test_positive_img_names    );
        load_names( full_test_neg,   test+"/",   full_test,  test_negative_img_names    );

        load_images(train_positive_img_names, train_positive_imgs);
        load_images(train_negative_img_names, train_negative_imgs);
        load_images(test_positive_img_names, test_positive_imgs);
        load_images(test_negative_img_names, test_negative_imgs);

        auto train_samples =    train_positive_imgs.size() +
                                (windows_per_negative_sample.width * windows_per_negative_sample.height) * train_negative_imgs.size();
        auto test_pos_samples = test_positive_imgs.size();
        auto test_neg_samples = (windows_per_negative_sample.width * windows_per_negative_sample.height) * test_negative_imgs.size();
        //420 is a constant given by the size of the window...
        //HOG is usually using a feature vector = #Blocks * #CellsPerBlock * #BinsPerCell = 3780
        int feature_size =      420 * dct_selection_mask.bins * (dct_selection_mask.use_pos_neg ? 2 : 1);

        train_features.create(cv::Size(feature_size, (int)train_samples), CV_32FC1);
        train_labels.create(cv::Size(1, (int)(train_samples)), CV_32FC1);

        test_pos_features.create(cv::Size(feature_size, (int)test_pos_samples), CV_32FC1);
        expected_pos_test_labels.create(cv::Size(1, (int)(test_pos_samples)), CV_32FC1);
        actual_pos_test_labels.create(cv::Size(1, (int)(test_pos_samples)), CV_32FC1);

        test_neg_features.create(cv::Size(feature_size, (int)test_neg_samples), CV_32FC1);
        expected_neg_test_labels.create(cv::Size(1, (int)(test_neg_samples)), CV_32FC1);
        actual_neg_test_labels.create(cv::Size(1, (int)(test_neg_samples)), CV_32FC1);

        //sample the actual windows

        int train_idx{0};
        //positive train
        std::cout<<"positive train windows: ";
        std::for_each(train_positive_imgs.begin(), train_positive_imgs.end(),
                      [this, &train_idx](cv::Mat& image){
                          extract_middle_window(image, dct_selection_mask, train_window_stride, train_features, train_idx);
                      });
        std::cout<<train_idx<<std::endl;

        //negative train
        auto train_idx_tmp = train_idx;
        std::cout<<"negative train windows: ";
        std::for_each(train_negative_imgs.begin(), train_negative_imgs.end(),
                      [this, &train_idx](cv::Mat& image){
                          extract_matrix_windows(image, dct_selection_mask, windows_per_negative_sample, train_features, train_idx);
                      });
        std::cout<<train_idx-train_idx_tmp<<std::endl;

        //set the labels
        cv::Mat hdr1(train_labels, cv::Rect(cv::Point(0,0), cv::Size(1, train_idx_tmp)));
        hdr1.setTo(-1.f);
        cv::Mat hdr2(train_labels, cv::Rect(cv::Point(0,train_idx_tmp), cv::Size(1, train_idx-train_idx_tmp)));
        hdr2.setTo(1.f);

        int test_idx{0};
        //positive test
        std::cout<<"positive test windows: ";
        std::for_each(test_positive_imgs.begin(), test_positive_imgs.end(),
                      [this, &test_idx](cv::Mat& image){
                          extract_middle_window(image, dct_selection_mask, test_window_stride, test_pos_features, test_idx);
                      });
        std::cout<<test_idx<<std::endl;

        //set the labels
        expected_pos_test_labels.setTo(-1.f);
        //actual_pos_test_labels.setTo(-1.f);

        //negative test
        test_idx = 0;
        std::cout<<"negative test windows: ";
        std::for_each(test_negative_imgs.begin(), test_negative_imgs.end(),
                      [this, &test_idx](cv::Mat& image){
                          extract_matrix_windows(image, dct_selection_mask, windows_per_negative_sample, test_neg_features, test_idx);
                      });
        std::cout<<test_idx<<std::endl;

        expected_neg_test_labels.setTo(1.f);
        //actual_neg_test_labels.setTo(1.f);

        std::cout   << std::endl
                    <<"train windows: "<<train_idx<<std::endl
                    <<"test windows: "<<test_idx<<std::endl;

        std::cout<<"end loading data"<<std::endl;
    }

    void run_hof_test() {
        std::cout<<"start training"<<std::endl;

        CvSVMParams params;
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-6);

        CvSVM SVM;
        //SVM.train(train_features, train_labels, cv::Mat(), cv::Mat(), params);
        SVM.train_auto(train_features, train_labels, cv::Mat(), cv::Mat(), params, 10);

        std::cout<<"end training"<<std::endl;

        std::cout<<"start testing"<<std::endl;
        SVM.predict(test_pos_features, actual_pos_test_labels);
        std::cout << "Pos:" << cv::countNonZero(actual_pos_test_labels != expected_pos_test_labels) << std::endl;
        SVM.predict(test_neg_features, actual_neg_test_labels);
        std::cout << "Neg:" << cv::countNonZero(actual_neg_test_labels != expected_neg_test_labels) << std::endl;
        std::cout<<"end testing"<<std::endl;
    }

    void run_hog_test(){
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

        uint64_t cnt = 0;
        std::for_each(train_positive_imgs.begin(), train_positive_imgs.end(), 
            [&hog, &cnt](cv::Mat& img){
                std::vector<cv::Rect> locations;
                hog.detectMultiScale(img, locations, 0, cv::Size(2,2), cv::Size(16,16), 1.01);
                cnt += locations.size() > 0 ? 0 : 1;

                /*
                hog.detectMultiScale(img, locations, 0, cv::Size(2,2), cv::Size(16,16), 1.01);

                3634
                HOG FN: 751
                OK...
                */
            });

        std::cout << train_positive_imgs.size() << std::endl;
        std::cout << "HOG FN: " << cnt << std::endl;
    }
};

int main(int argc, const char* argv[]){
    try{
        if (argc != 7)
            throw std::invalid_argument("invalid number of arguments, correct formar: ./experiment dataset_dir selection_mask_file neg_w_samples neg_h_samples use_pos_neg use_abs_val");

        experiment_t exp{
            check_directory(argv[1]),               // INRIA dataset
            check_file(argv[2]),                    // dct selection mask
            check_samples(std::stoi(argv[3])),      // negative samples per w
            check_samples(std::stoi(argv[4])),      // negative samples per h

            (bool)std::stoi(argv[5]),               // use pos/neg
            (bool)std::stoi(argv[6])                // use abs
        };

        exp.load_data();
        exp.run_hof_test();

        /*
        ./INRIAPerson ./selection_window.txt 8 4 1 1
        start loading data
        positive train windows: 2416
        negative train windows: 38976
        positive test windows: 1132
        negative test windows: 14496

        train windows: 41392
        test windows: 14496
        end loading data
        start training
        end training
        start testing
        Pos:134
        Neg:956
        end testing
        OK...
        */

        std::cout << "OK..." << std::endl;
        cv::waitKey();
    } catch(const std::exception& e){
        std::cout << e.what() << std::endl;
    }

    return EXIT_SUCCESS;
}
