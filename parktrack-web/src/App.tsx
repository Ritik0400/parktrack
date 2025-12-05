import { useState, useEffect, useRef } from "react";
import axios from "axios";

type Detection = {
  bbox: [number, number, number, number];
  color?: string;
  color_confidence?: number;
  position?: string;
  matched_plate?: string | null;
  matched_score?: number | null;
  is_blacklisted?: boolean;   // NEW FIELD
  top_matches?: any[];
};

type DashboardEntry = {
  plate: string;
  last_seen: string;
  lot: string;
  owner_name: string | null;
  car_model: string | null;
};

type DashboardResponse = {
  total: number;
  lots: { A: number; B: number; C: number };
  entries: DashboardEntry[];
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const [page, setPage] = useState<"identify" | "dashboard">("identify");
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [isDashLoading, setIsDashLoading] = useState(false);

  const [showBlacklistPopup, setShowBlacklistPopup] = useState<boolean>(false);
  const [blacklistedPlate, setBlacklistedPlate] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgDims, setImgDims] = useState<{
    naturalW: number;
    naturalH: number;
    displayW: number;
    displayH: number;
  } | null>(null);

  // ------------------------------------------------------
  // Image Upload
  // ------------------------------------------------------
  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;

    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setImgDims(null);
  };

  const onImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImgDims({
      naturalW: img.naturalWidth,
      naturalH: img.naturalHeight,
      displayW: img.clientWidth,
      displayH: img.clientHeight,
    });
  };

  // ------------------------------------------------------
  // IDENTIFY CAR API
  // ------------------------------------------------------
  const identifyCar = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("lot", "A");

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/api/v1/reid/identify-cars",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(res.data);

      // --------------------------------------------------
      // CHECK BLACKLIST AFTER DETECTIONS
      // --------------------------------------------------
      const detections: Detection[] = res.data?.detections ?? [];
      const matched = detections.filter(
        (d) => d.matched_plate && d.matched_score !== null
      );

      const blacklisted = matched.find((d) => d.is_blacklisted === true);

      if (blacklisted) {
        setBlacklistedPlate(blacklisted.matched_plate || "Unknown");
        setShowBlacklistPopup(true);
      }
    } catch (err) {
      console.error(err);
      setError("Car detection failed. Check backend logs.");
    }

    setIsLoading(false);
  };

  const detections: Detection[] = result?.detections ?? [];

  const matchedDetections = detections.filter(
    (d) => d.matched_plate && d.matched_score !== null
  );
  const isSuccess = matchedDetections.length > 0;

  const scaleBox = (bbox: [number, number, number, number]) => {
    if (!imgDims) return { left: 0, top: 0, width: 0, height: 0 };

    const [x, y, w, h] = bbox;
    const sx = imgDims.displayW / imgDims.naturalW;
    const sy = imgDims.displayH / imgDims.naturalH;

    return {
      left: x * sx,
      top: y * sy,
      width: w * sx,
      height: h * sy,
    };
  };

  // ------------------------------------------------------
  // DASHBOARD API
  // ------------------------------------------------------
  const fetchDashboard = async () => {
    setIsDashLoading(true);
    try {
      const res = await axios.get("http://127.0.0.1:8000/api/v1/dashboard");
      setDashboard(res.data);
    } catch (err) {
      console.error(err);
    }
    setIsDashLoading(false);
  };

  useEffect(() => {
    if (page !== "dashboard") return;
    fetchDashboard();
    const interval = setInterval(fetchDashboard, 5000);
    return () => clearInterval(interval);
  }, [page]);

  // ------------------------------------------------------
  // RENDER UI
  // ------------------------------------------------------
  return (
    <div className="min-h-screen p-5" style={{ backgroundColor: "#d6eaff" }}>
      <h1 className="text-3xl font-bold text-center mb-6">
        ParkTrack Car Identification System
      </h1>

      {/* Navigation */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          className={`px-6 py-3 rounded-lg font-semibold ${
            page === "identify"
              ? "bg-blue-600 text-white"
              : "bg-gray-300 text-black"
          }`}
          onClick={() => setPage("identify")}
        >
          Identify Car
        </button>

        <button
          className={`px-6 py-3 rounded-lg font-semibold ${
            page === "dashboard"
              ? "bg-blue-600 text-white"
              : "bg-gray-300 text-black"
          }`}
          onClick={() => setPage("dashboard")}
        >
          Live Dashboard
        </button>
      </div>

      {/* --------------------------------------------------
           IDENTIFY PAGE
      -------------------------------------------------- */}
      {page === "identify" && (
        <>
          <div className="flex flex-col items-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleUpload}
              className="mb-4"
            />

            {preview && (
              <div className="relative inline-block rounded-lg shadow-lg bg-white p-3">
                <img
                  ref={imgRef}
                  src={preview}
                  alt="Uploaded"
                  onLoad={onImageLoad}
                  className="max-w-3xl max-h-[480px] rounded"
                />

                {/* Bounding Boxes */}
                {imgDims &&
                  detections.map((det, idx) => {
                    const { left, top, width, height } = scaleBox(det.bbox);
                    const hasMatch = !!det.matched_plate;

                    const borderColor = hasMatch
                      ? "border-green-400"
                      : "border-yellow-300";

                    const bgLabel = hasMatch
                      ? "bg-green-500"
                      : "bg-yellow-400";

                    return (
                      <div
                        key={idx}
                        className={`absolute border-2 ${borderColor} rounded`}
                        style={{
                          left,
                          top,
                          width,
                          height,
                          pointerEvents: "none",
                        }}
                      >
                        <div
                          className={`${bgLabel} text-xs text-white px-1 py-0.5 rounded-br`}
                        >
                          {det.position?.toUpperCase()}{" "}
                          {hasMatch ? `• ${det.matched_plate}` : ""}
                          {det.is_blacklisted ? " ⚠" : ""}
                        </div>
                      </div>
                    );
                  })}
              </div>
            )}

            <button
              onClick={identifyCar}
              disabled={isLoading}
              className={`px-6 py-3 rounded-lg text-white font-semibold mt-4 ${
                isLoading ? "bg-gray-400" : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {isLoading ? "Identifying..." : "Identify Car"}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="mt-6 bg-red-200 text-red-900 p-4 rounded-lg text-center max-w-xl mx-auto">
              {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div
              className={`mt-8 p-6 rounded-lg shadow-lg max-w-2xl mx-auto text-white ${
                isSuccess ? "bg-green-600" : "bg-red-600"
              }`}
            >
              <h2 className="text-2xl font-bold mb-3">
                {isSuccess
                  ? `High-confidence matches (≥95%): ${matchedDetections.length}`
                  : "No high-confidence matches (≥95%)"}
              </h2>

              <p className="text-sm mb-4">
                Lot: <b>{result.lot}</b> • Total detections:{" "}
                <b>{detections.length}</b>
              </p>

              {matchedDetections.map((det, idx) => (
                <div key={idx} className="mb-4 bg-white text-black p-4 rounded-lg">
                  <p className="font-bold text-lg">
                    Matched Plate: {det.matched_plate}
                    {det.is_blacklisted && (
                      <span className="text-red-600 font-bold ml-2">
                        ⚠ BLACKLISTED
                      </span>
                    )}
                  </p>

                  <p>Confidence: {det.matched_score?.toFixed(1)}%</p>
                  <p>Position: {det.position}</p>
                  <p>
                    Color: {det.color} (conf{" "}
                    {(det.color_confidence! * 100).toFixed(1)}%)
                  </p>

                  {det.top_matches?.length > 0 && (
                    <div className="mt-3 p-3 bg-gray-100 rounded-lg">
                      <p className="font-semibold mb-1">
                        Car Details (best match):
                      </p>
                      <p>Owner: {det.top_matches[0].meta.owner_name}</p>
                      <p>Model: {det.top_matches[0].meta.car_model}</p>
                      <p>Plate: {det.top_matches[0].meta.plate}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* --------------------------------------------------
           DASHBOARD PAGE
      -------------------------------------------------- */}
      {page === "dashboard" && (
        <div className="w-full max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold">Live Parking Dashboard</h2>
            <button
              onClick={fetchDashboard}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg"
            >
              Refresh
            </button>
          </div>

          {isDashLoading && <p className="text-gray-600">Loading…</p>}

          {dashboard && (
            <>
              <p className="mb-3">
                Total cars seen today: <b>{dashboard.total}</b> | Lot A:{" "}
                <b>{dashboard.lots.A}</b> | Lot B: <b>{dashboard.lots.B}</b> |
                Lot C: <b>{dashboard.lots.C}</b>
              </p>

              <table className="w-full border mt-4">
                <thead className="bg-gray-200">
                  <tr>
                    <th className="p-2 border">Plate</th>
                    <th className="p-2 border">Last Seen</th>
                    <th className="p-2 border">Lot</th>
                    <th className="p-2 border">Owner</th>
                    <th className="p-2 border">Model</th>
                  </tr>
                </thead>

                <tbody>
                  {dashboard.entries.map((e, idx) => (
                    <tr key={idx} className="text-center">
                      <td className="p-2 border font-semibold">{e.plate}</td>
                      <td className="p-2 border">
                        {new Date(e.last_seen).toLocaleString()}
                      </td>
                      <td className="p-2 border">{e.lot}</td>
                      <td className="p-2 border">{e.owner_name}</td>
                      <td className="p-2 border">{e.car_model}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}

      {/* --------------------------------------------------
          BLACKLIST POPUP MODAL
      -------------------------------------------------- */}
      {showBlacklistPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-xl max-w-md text-center">
            <h2 className="text-2xl font-bold text-red-600 mb-3">
              ⚠ BLACKLISTED VEHICLE DETECTED
            </h2>
            <p className="text-lg mb-4">
              The vehicle with plate <b>{blacklistedPlate}</b> is flagged as
              blacklisted.
            </p>

            <button
              onClick={() => setShowBlacklistPopup(false)}
              className="px-5 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
