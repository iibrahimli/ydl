/*! @file

    The network class that encapsulates a Darknet network
    A wrapper for the darknet library written by Joseph Redmon
    (https://github.com/pjreddie/darknet)

*/


#ifndef YDL_H_
#define YDL_H_

#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <tuple>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "darknet.h"



namespace ydl{
    
    /// time interval
    using duration = std::chrono::high_resolution_clock::duration;


    /// vector of strings (used to store class names)
    using v_str = std::vector<std::string>;


    /// vector of colors (used for visualization of bboxes)
    using v_col = std::vector<cv::Scalar>;


    /*! 
        Map of a class id to a probability that this object belongs to that class.
        The key is the zero-based index of the class, while the value is the probability
        that the object belongs to that class.
    */
    using m_class_prob = std::map<int, float>;


    /// Structure that stores information on predictions
    struct pred_result {

        /// Opencv rectangle which describes where the object is located on the image
        cv::Rect rect;

        /// Normalized x coordinate of the center of the bbox (use rect.x instead)
        float mid_x;

        /// Normalized y coordinate of the center of the bbox (use rect.y instead)    
        float mid_y;

        /// Normalized width of the bbox (use rect.width instead)
        float width;

        /// Normalized width of the bbox (use rect.height instead)
        float height;

        /// Map of ids to probabilities for objects predicted with non-zero probability
        m_class_prob all_prob;

        /// id of maximum probability class
        int best_class;

        /// the probability of the class that obtained the highest value
        float best_prob;

        /// the name of highest probability class used to annotate the object
        std::string name;

    };


    /// vector of prediction results
    using v_pred_result = std::vector<pred_result>;


    class detector;


} // namespace azc




/*!
    The main class. Encapsulates a Darknet network, is used
    to predict on an image, video, opencv stream, etc.
*/
class ydl::detector {

public:

    /// Destructor
    virtual ~detector();


    /// Constructor
    detector(const std::string& cfg_filename, const std::string& weights_filename, const std::string& names_filename = "");


    /*!
        Use network to predict on an image.
        @param image_filename The image to predict on. The member original_image will be set to this image.
        @param threshold If <0, previous threshold will be used.
        @returns A vector of prediction results.
    */
    virtual std::tuple<v_pred_result, duration> predict(const std::string& image_filename, float threshold = -1.0f);


    /*!
        Use network to predict on an image.
        @param mat An opencv image which has already been loaded. The member original_image will be set to this image.
        @param threshold If <0, previous threshold will be used.
        @returns A vector of prediction results.
    */
    virtual std::tuple<v_pred_result, duration> predict(cv::Mat mat, float threshold = -1.0f);


    /*!
        Use network to predict on an image.
        @param img A Darknet image which has already been loaded. The member original_image will be set to this image.
        @param threshold If <0, previous threshold will be used.
        @returns A vector of prediction results.
    */
    virtual std::tuple<v_pred_result, duration> predict(image img, float threshold = -1.0f);


    /*!
        Static function to convert the OpenCV @p cv::Mat objects to Darknet's internal @p image format.
        Provided for convenience in case you need to call into one of Darknet's functions.
        @see @ref convert_darknet_image_to_opencv_mat()
    */
    static image convert_opencv_mat_to_darknet_image(cv::Mat mat);


    /*!
        Static function to convert Darknet's internal @p image format to OpenCV's @p cv::Mat format.
        Provided for convenience in case you need to manipulate a Darknet image.
        @see @ref convert_opencv_mat_to_darknet_image()
    */
    static cv::Mat convert_darknet_image_to_opencv_mat(const image img);


    /*!
        Static function to annotate an image using predictions
    */
    static cv::Mat annotate(const cv::Mat& mat, ydl::v_pred_result pr);


    /*!
        The Darknet network. This is setup in the constructor.
        @note Unfortunately, the Darknet C API does not allow this to be de-allocated!
    */
    network *net;


    /// Names of classes
    v_str names;


    ///
    v_col colors;


    /// The time it took to load the network weights 
    duration load_duration;


    /// Image prediction threshold. Defaults to 0.5.  @see @ref predict()
    float threshold;


    /*!
        Used during prediction. Defaults to 0.5. @see @ref predict()
        @todo Need to find more details on how this setting works in Darknet.
    */
    float hier_threshold;


    /*!
        Non-Maximal Suppression (NMS) threshold suppresses overlapping bounding boxes and only retains the bounding
        box that has the maximum probability of object detection associated with it. Defaults to 0.45.
        @see @ref predict()
     */
    float nms_threshold;


    // annotation parameters
    cv::HersheyFonts  annotation_font_face;
	double            annotation_font_scale;
	int               annotation_font_thickness;
	bool              annotation_include_duration;
	bool              annotation_include_timestamp;
	bool              names_include_percentage;

};



/// print detection result for debugging purposes
std::ostream & operator<<(std::ostream & os, const ydl::pred_result & pred);


/// print a vector of prediction results
std::ostream & operator<<(std::ostream & os, const ydl::v_pred_result & results);



#endif // YDL_H_