#include "detector.hpp"

namespace ydl{



detector::~detector(){
}


detector::detector(const std::string& cfg_filename, const std::string& weights_filename, const std::string& names_filename){

    // load network
    const auto t1 = std::chrono::high_resolution_clock::now();
	net = load_network(const_cast<char*>(cfg_filename.c_str()), const_cast<char*>(weights_filename.c_str()), 1);
	const auto t2 = std::chrono::high_resolution_clock::now();
	load_duration = t2 - t1;

	std::cout << "loaded weights in " << std::chrono::duration_cast<std::chrono::milliseconds>(load_duration).count() << " ms" << std::endl;

    if (net == nullptr)
	{
		/// @throw std::runtime_error if the call to darknet's load_network() has failed.
		throw std::runtime_error("darknet failed to load the configuration, the weights, or both");
	}

    set_batch_network(net, 1);

	// set properties
	threshold       = 0.25f;
	hier_threshold  = 0.5f;
	nms_threshold   = 0.45f;

	// parse names
	if (! names_filename.empty()){
		std::ifstream ifs(names_filename);
		std::string line;
		while (std::getline(ifs, line)){
			if (line.empty())
				break;
			names.push_back(line);
		}
	}

}


// initialize static variables
cv::HersheyFonts  detector::annotation_font_face        = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
double            detector::annotation_font_scale       = 0.5;
int               detector::annotation_font_thickness   = 1;
bool              detector::annotation_include_duration = true;
bool              detector::names_include_percentage    = true;
v_col             detector::annot_colors                = get_default_annotation_colors();



std::tuple<v_pred_result, duration> detector::predict(const std::string& image_filename, float threshold){
	
	cv::Mat mat = cv::imread(image_filename);

	if (mat.empty()){
		/// @throw std::invalid_argument if the image failed to load.
		throw std::invalid_argument("failed to load image \"" + image_filename + "\"");
	}

	return predict(mat, threshold);
}



std::tuple<v_pred_result, duration> detector::predict(cv::Mat mat, float threshold){

	v_pred_result result;

	if (net == nullptr){
		/// @throw std::logic_error if the network is invalid.
		throw std::logic_error("cannot predict with an empty network");
	}

	// user has probably specified percentages, so bring it back down to a range between 0.0 and 1.0
	if (threshold > 1.0) threshold /= 100.0;
	if (threshold < 0.0) threshold  = 0.1;
	if (threshold > 1.0) threshold  = 1.0;
	
	cv::Mat resized_image;
	cv::resize(mat, resized_image, cv::Size(net->w, net->h));
	image img = convert_opencv_mat_to_darknet_image(resized_image);

	float * X = img.data;

	const auto t1 = std::chrono::high_resolution_clock::now();
	network_predict(net, X);
	const auto t2 = std::chrono::high_resolution_clock::now();
	duration duration = t2 - t1;

	int nboxes = 0;
	auto darknet_results = get_network_boxes(net, 1, 1, threshold, hier_threshold, 0, 1, &nboxes);

	if(nms_threshold){
		auto net_layer = net->layers[net->n - 1];
		do_nms_sort(darknet_results, nboxes, net_layer.classes, nms_threshold);		
		// do_nms_obj(darknet_results, nboxes, net_layer.classes, nms_threshold);
	}

	for (int detection_idx = 0; detection_idx < nboxes; detection_idx ++){

		const auto & det = darknet_results[detection_idx];

		// NMS will set objectness of redundant boxes to 0
		if(det.objectness == 0.)
			continue;

		if (names.empty()){
			// we weren't given a names file to parse, but we know how many classes are defined in the network
			// so we can use dummy names based on the class index
			for (int i = 0; i < det.classes; i++)
				names.push_back("class_" + std::to_string(i));
		}

		/*
			The "det" object has an array called det.prob[].  That array is large enough for 1 entry per class in the network.
			Each entry will be set to 0.0, except for the ones that correspond to the class that was detected. Note that it
			is possible that multiple entries are non-zero!  We need to look at every entry and remember which ones are set.
		*/
		pred_result pr;
		pr.best_class = 0;
		pr.best_prob = 0.0f;

		for (int class_idx = 0; class_idx < det.classes; ++class_idx){
			if (det.prob[class_idx] >= threshold){
				// remember this probability since it is higher than the threshold
				pr.all_prob[class_idx] = det.prob[class_idx];

				// see if this is the highest probability we've seen
				if (det.prob[class_idx] > pr.best_prob){
					pr.best_class = class_idx;
					pr.best_prob  = det.prob[class_idx];
				}
			}
		}

		if (pr.best_prob >= threshold){
			// at least 1 class is beyong the threshold, so remember this object

			/*
				int left  = (b.x-b.w/2.)*im.w;
				int right = (b.x+b.w/2.)*im.w;
				int top   = (b.y-b.h/2.)*im.h;
				int bot   = (b.y+b.h/2.)*im.h;
			*/

			const int w = std::round(det.bbox.w * mat.cols);
			const int h = std::round(det.bbox.h * mat.rows);
			const int x = std::round(det.bbox.x * mat.cols - w/2.);
			const int y = std::round(det.bbox.y * mat.rows - h/2.);
			pr.rect = cv::Rect(cv::Point(x, y), cv::Size(w, h));

			pr.mid_x	= det.bbox.x;
			pr.mid_y	= det.bbox.y;
			pr.width	= det.bbox.w;
			pr.height	= det.bbox.h;

			pr.name = names.at(pr.best_class);

			if(names_include_percentage){
				const int percentage = std::round(100.0 * pr.best_prob);
				pr.name += " " + std::to_string(percentage) + "%";
			}

			result.push_back(pr);
		}
	}

	free_detections(darknet_results, nboxes);
	free_image(img);

	return {result, duration};
}



std::tuple<v_pred_result, duration> detector::predict(image img, float threshold){
	cv::Mat mat = convert_darknet_image_to_opencv_mat(img);

	return predict(mat, threshold);
}



image detector::convert_opencv_mat_to_darknet_image(cv::Mat mat){
// this function is taken/inspired directly from Darknet:  image_opencv.cpp, mat_to_image()

	// OpenCV uses BGR, but Darknet expects RGB
	if (mat.channels() == 3){
		cv::Mat rgb;
		cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
		mat = rgb;
	}

	const int width		= mat.cols;
	const int height	= mat.rows;
	const int channels	= mat.channels();
	const int step		= mat.step;
	image img			= make_image(width, height, channels);
	uint8_t * data		= (uint8_t*)mat.data;

	for (int y = 0; y < height; ++y){
		for (int c = 0; c < channels; ++c){
			for (int x = 0; x < width; ++x){
				img.data[c*width*height + y*width + x] = data[y*step + x*channels + c] / 255.0f;
			}
		}
	}

	return img;
}



cv::Mat detector::convert_darknet_image_to_opencv_mat(const image img){
	// this function is taken/inspired directly from Darknet:  image_opencv.cpp, image_to_mat()

	const int channels	= img.c;
	const int width		= img.w;
	const int height	= img.h;
	cv::Mat mat			= cv::Mat(height, width, CV_8UC(channels));
	const int step		= mat.step;

	for (int y = 0; y < height; ++y){
		for (int x = 0; x < width; ++x){
			for (int c = 0; c < channels; ++c){
				float val = img.data[c*height*width + y*width + x];
				mat.data[y*step + x*channels + c] = (unsigned char)(val * 255);
			}
		}
	}

	// But now the mat is in RGB instead of the BGR format that OpenCV expects to use.  See show_image_cv()
	// in Darknet which does the RGB<->BGR conversion, which we'll copy here so the mat is immediately usable.
	if (channels == 3)
		cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

	return mat;
}



cv::Mat detector::annotate(cv::Mat& mat, v_pred_result pr, duration dur){
	if (mat.empty()){
		/// @throw std::logic_error if an attempt is made to annotate an empty image
		throw std::logic_error("cannot annotate an empty image");
	}

	// for fps string
	std::stringstream ss;

	// cv::Mat annotated_mat = mat.clone();
	cv::Mat annotated_mat = mat;

	// make sure we always have colours we can use
	if (annot_colors.empty()){
		annot_colors = get_default_annotation_colors();
	}

	for (const auto & pred : pr){
		const auto color = annot_colors[pred.best_class % annot_colors.size()];

//			std::cout << "class id=" << pred.best_class << ", probability=" << pred.best_probability << ", point=(" << pred.rect.x << "," << pred.rect.y << "), name=\"" << pred.name << "\", duration=" << duration_string() << std::endl;
		cv::rectangle(annotated_mat, pred.rect, color, 2);

		const cv::Size text_size = cv::getTextSize(pred.name, annotation_font_face, annotation_font_scale, annotation_font_thickness, nullptr);

		cv::Rect r(cv::Point(pred.rect.x - 1, pred.rect.y - text_size.height - 2), cv::Size(text_size.width + 2, text_size.height + 2));
		cv::rectangle(annotated_mat, r, color, CV_FILLED);
		cv::putText(annotated_mat, pred.name, cv::Point(r.x + 1, r.y + text_size.height), annotation_font_face, annotation_font_scale, cv::Scalar(0,0,0), annotation_font_thickness, CV_AA);
	}

	if (annotation_include_duration){
		std::string str		= duration_string(dur);
		
		// get fps up to 2 decimal places
		auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		float fps = (float) 1000 / millis;
		ss.clear();
		ss << std::setw(6) << std::fixed << std::setprecision(2) << fps;
		std::string fps_str = ss.str();
		str = str + " (" + fps_str + " fps)";
		
		// draw inference time and FPS
		const cv::Size text_size	= cv::getTextSize(str, annotation_font_face, annotation_font_scale, annotation_font_thickness, nullptr);

		// draw label and confidence
		cv::Rect r(cv::Point(2, 2), cv::Size(text_size.width + 2, text_size.height + 2));
		cv::rectangle(annotated_mat, r, cv::Scalar(255,255,255), CV_FILLED);
		cv::putText(annotated_mat, str, cv::Point(r.x + 1, r.y + text_size.height), annotation_font_face, annotation_font_scale, cv::Scalar(0,0,0), annotation_font_thickness, CV_AA);
	}

	return annotated_mat;
}


v_col detector::get_default_annotation_colors(){
	v_col colors =
	{
		// blue, green, red
		{0x5E, 0x35, 0xFF},	// Radical Red
		{0x17, 0x96, 0x29},	// Slimy Green
		{0x33, 0xCC, 0xFF},	// Sunglow
		{0x4D, 0x6E, 0xAF},	// Brown Sugar
		{0xFF, 0x00, 0xFF},	// magenta
		{0xE6, 0xBF, 0x50},	// Blizzard Blue
		{0x00, 0xFF, 0xCC},	// Electric Lime
		{0xFF, 0xFF, 0x00},	// cyan
		{0x85, 0x4E, 0x8D},	// Razzmic Berry
		{0xCC, 0x00, 0xFF},	// Purple Pizzazz
		{0x00, 0xFF, 0x00},	// green
		{0x00, 0xFF, 0xFF},	// yellow
		{0xEC, 0xAD, 0x5D},	// Blue Jeans
		{0xFF, 0x6E, 0xFF},	// Shocking Pink
		{0x66, 0xFF, 0xFF},	// Laser Lemon
		{0xD1, 0xF0, 0xAA},	// Magic Mint
		{0x00, 0xC0, 0xFF},	// orange
		{0xB6, 0x51, 0x9C},	// Purple Plum
		{0x33, 0x99, 0xFF},	// Neon Carrot
		{0xFF, 0x00, 0xFF},	// blue
		{0x66, 0xFF, 0x66},	// Screamin' Green
		{0x00, 0x00, 0xFF},	// red
		{0x37, 0x60, 0xFF},	// Outrageous Orange
		{0x78, 0x5B, 0xFD}	// Wild Watermelon
	};

	return colors;
}



std::string detector::duration_string(duration dur){
	std::string str;
	if		(dur <= std::chrono::nanoseconds(1000))  { str = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count())  + " ns"; }
	else if	(dur <= std::chrono::microseconds(1000)) { str = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(dur).count()) + " us"; }
	else if	(dur <= std::chrono::milliseconds(1000)) { str = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()) + " ms"; }
	else /* use milliseconds for anything longer */	 { str = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()) + " ms"; }

	return str;
}



}  // namespace ydl



std::ostream & operator<<(std::ostream & os, const ydl::pred_result & pred){
	os << "\""			<< pred.name << "\""
	   << " #"			<< pred.best_class
	   << " prob="		<< pred.best_prob
	   << " x="		    << pred.rect.x
	   << " y="		    << pred.rect.y
	   << " w="		    << pred.rect.width
	   << " h="		    << pred.rect.height
	   << " entries="	<< pred.all_prob.size();

	if (pred.all_prob.size() > 1){
		os << " [";
		for (auto iter : pred.all_prob){
			const auto & key = iter.first;
			const auto & val = iter.second;
			os << " " << key << ":" << val;
		}
		os << " ]";
	}

	return os;
}


std::ostream & operator<<(std::ostream & os, const ydl::v_pred_result & results){
	const size_t number_of_results = results.size();
	os << "prediction results: " << number_of_results;

	for (size_t idx = 0; idx < number_of_results; idx ++){
		os << std::endl << "-> " << (idx+1) << "/" << number_of_results << ": ";
		operator<<(os, results.at(idx));
	}

	return os;
}