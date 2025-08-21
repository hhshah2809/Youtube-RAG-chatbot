import React, { useState } from "react";

function LeafCheck() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setResult(null); // Clear previous result when a new image is selected
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) {
      alert("Please upload an image!");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);
      const response = await fetch("http://127.0.0.1:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.result);
      } else {
        setResult({ error: "Prediction failed" });
      }
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Something went wrong!" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white shadow rounded">
      <h1 className="text-2xl font-bold mb-4 text-center">
        Leaf Freshness Check
      </h1>

      <form onSubmit={handleSubmit} className="flex flex-col gap-2">
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button
          type="submit"
          className="bg-green-600 text-white px-4 py-2 rounded"
          disabled={loading}
        >
          {loading ? "Checking..." : "Upload & Check"}
        </button>
      </form>

      {result && (
        <div className="mt-4 text-lg">
          {/* Error case */}
          {result.error ? (
            <p className="text-red-500 font-semibold">{result.error}</p>
          ) : !result.ok ? (
            // Backend said "no leaf detected"
            <p className="text-red-500 font-semibold">
              {result.reason || "Leaf not detected but high chances of non-fresh"}
            </p>
          ) : (
            // Normal case with features
            <>
              <p className="font-semibold">
                Freshness: {result.fresh ? "Fresh" : "Not Fresh"}
              </p>
              <p>Probability: {(result.proba * 100).toFixed(2)}%</p>

              {result.features && (
                <>
                  <h2 className="mt-2 font-semibold">Features:</h2>
                  <ul className="list-disc ml-5">
                    <li>GCV: {result.features.gcv.toFixed(2)}</li>
                    <li>Area: {result.features.area.toFixed(2)}</li>
                    <li>
                      Aspect Ratio: {result.features.aspect_ratio.toFixed(2)}
                    </li>
                    <li>
                      Roundness: {result.features.roundness.toFixed(2)}
                    </li>
                  </ul>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default LeafCheck;
