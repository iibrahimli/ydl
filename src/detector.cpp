#include "detector.hpp"



ydl::detector::detector(const std::string& cfg_filename, const std::string& weights_filename, const std::string& names_filename = ""){

    // load network
    const auto t1 = std::chrono::high_resolution_clock::now();
	net = load_network(const_cast<char*>(cfg_filename.c_str()), const_cast<char*>(weights_filename.c_str()), 1);
	const auto t2 = std::chrono::high_resolution_clock::now();
	load_duration = t2 - t1;

    if (net == nullptr)
	{
		/// @throw std::runtime_error if the call to darknet's load_network() has failed.
		throw std::runtime_error("darknet failed to load the configuration, the weights, or both");
	}

    set_batch_network(net, 1);

	// set properties
	threshold							= 0.5f;
	hier_threshold			     		= 0.5f;
	nms_threshold	                    = 0.45f;

	annotation_font_face				= cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
	annotation_font_scale				= 0.5;
	annotation_font_thickness			= 1;
	annotation_include_duration			= true;
	names_include_percentage			= true;

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



std::tuple<ydl::v_pred_result, ydl::duration> ydl::detector::predict(const std::string& image_filename, float threshold){
	
	cv::Mat mat = cv::imread(image_filename);

	if (mat.empty()){
		/// @throw std::invalid_argument if the image failed to load.
		throw std::invalid_argument("failed to load image \"" + image_filename + "\"");
	}

	return predict(mat, threshold);
}



std::tuple<ydl::v_pred_result, ydl::duration> ydl::detector::predict(cv::Mat mat, float threshold){

	v_pred_result result;

	if (net == nullptr){
		/// @throw std::logic_error if the network is invalid.
		throw std::logic_error("cannot predict with an empty network");
	}

	// user has probably specified percentages, so bring it back down to a range between 0.0 and 1.0
	if (threshold > 1.0) threshold /= 100.0;
	if (threshold < 0.0) threshold = 0.1;
	if (threshold > 1.0) threshold = 1.0;
	
	cv::Mat resized_image;
	cv::resize(mat, resized_image, cv::Size(net->w, net->h));
	image img = convert_opencv_mat_to_darknet_image(resized_image);

	float * X = img.data;

	const auto t1 = std::chrono::high_resolution_clock::now();
	network_predict(net, X);
	const auto t2 = std::chrono::high_resolution_clock::now();
	ydl::duration duration = t2 - t1;

	int nboxes = 0;
	auto darknet_results = get_network_boxes(net, mat.cols, mat.rows, threshold, hier_threshold, 0, 1, &nboxes);

	if(nms_threshold){
		auto net_layer = net->layers[net->n - 1];
		do_nms_sort(darknet_results, nboxes, net_layer.classes, nms_threshold);
	}

	for (int detection_idx = 0; detection_idx < nboxes; detection_idx ++){

		const auto & det = darknet_results[detection_idx];

		if (names.empty()){
			// we weren't given a names file to parse, but we know how many classes are defined in the network
			// so we can invest a few dummy names to use based on the class index
			for (int i = 0; i < det.classes; i++)
				names.push_back("#" + std::to_string(i));
		}

		/*
			The "det" object has an array called det.prob[].  That array is large enough for 1 entry per class in the network.
			Each entry will be set to 0.0, except for the ones that correspond to the class that was detected.  Note that it
			is possible that multiple entries are non-zero!  We need to look at every entry and remember which ones are set.
		*/
		pred_result pr;
		pr.best_class = 0;
		pr.best_prob = 0.0f;

		for (int class_idx = 0; class_idx < det.classes; class_idx ++){
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

			const int w = std::round(det.bbox.w * mat.cols);
			const int h = std::round(det.bbox.h * mat.rows);
			const int x = std::round(det.bbox.x * mat.cols - w/2.0);
			const int y = std::round(det.bbox.y * mat.rows - h/2.0);
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



std::tuple<ydl::v_pred_result, ydl::duration> ydl::detector::predict(image img, float threshold){

}



image ydl::detector::convert_opencv_mat_to_darknet_image(cv::Mat mat){

}



cv::Mat ydl::detector::convert_darknet_image_to_opencv_mat(const image img){

}



cv::Mat ydl::detector::annotate(const cv::Mat& mat, ydl::v_pred_result pr){

}



std::ostream & operator<<(std::ostream & os, const ydl::pred_result & pred){
	os	<< "\""			<< pred.name << "\""
		<< " #"			<< pred.best_class
		<< " prob="		<< pred.best_prob
		<< " x="		<< pred.rect.x
		<< " y="		<< pred.rect.y
		<< " w="		<< pred.rect.width
		<< " h="		<< pred.rect.height
		<< " entries="	<< pred.all_prob.size();

	if (pred.all_prob.size() > 1)
	{
		os << " [";
		for (auto iter : pred.all_prob)
		{
			const auto & key = iter.first;
			const auto & val = iter.second;
			os << " " << key << "=" << val;
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