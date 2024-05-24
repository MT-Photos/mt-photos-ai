//
//  websocket.swift
//  chinese_clip
//
//  Created by Yixuan Si on 5/24/24.
//
//

import UIKit
import Vision

func ocr(in image: UIImage) -> [String: Any] {
    var resultsDict = [String: Any]()

    // Ensure the image has a CGImage.
    guard let cgImage = image.cgImage else {
        resultsDict["error"] = "The image is not valid."
        return resultsDict
    }

    // Create a new image request handler.
    let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])

    // Create a new request to recognize text, specifying the completion handler.
    let request = VNRecognizeTextRequest(completionHandler: { (request, error) in
        if let error = error {
            resultsDict["error"] = "Text recognition error: \(error.localizedDescription)"
            return
        }

        // Process the results of the request.
        guard let observations = request.results as? [VNRecognizedTextObservation] else {
            resultsDict["error"] = "Failed to process the image."
            return
        }

        var recognizedTexts = [Any]()
        for observation in observations {
            let topCandidates = observation.topCandidates(1)
            if let recognizedText = topCandidates.first {
                let boundingBox = VNImageRectForNormalizedRect(observation.boundingBox, Int(image.size.width), Int(image.size.height))

                recognizedTexts.append([
                    "text": recognizedText.string,
                    "confidence": recognizedText.confidence,
                    "boundingBox": [
                        "x": boundingBox.origin.x,
                        "y": boundingBox.origin.y,
                        "width": boundingBox.size.width,
                        "height": boundingBox.size.height
                    ]
                ])
            }
        }

        resultsDict["recognizedTexts"] = recognizedTexts
    })

    // Set the recognition level to accurate for better accuracy.
    request.recognitionLevel = .accurate
    // Specify the languages to recognize.
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en"]

    // Perform the text-recognition request.
    do {
        try requestHandler.perform([request])
    } catch {
        resultsDict["error"] = "Unable to perform the requests: \(error)."
    }
    
    return resultsDict
}

