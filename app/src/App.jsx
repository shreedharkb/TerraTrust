import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup } from 'react-leaflet';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts';
import Papa from 'papaparse';
import { Leaf, Droplets, MapPin, Activity, ShieldCheck, AlertTriangle } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet Default Icon Issue
import L from 'leaflet';
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const App = () => {
  const [geoData, setGeoData] = useState(null);
  const [farmData, setFarmData] = useState([]);
  const [activeFarm, setActiveFarm] = useState(null);

  useEffect(() => {
    // Load GeoJSON Boundaries
    fetch('/data/davangere_taluks.geojson')
      .then((res) => res.json())
      .then((data) => setGeoData(data));

    // Load Credit Scores
    // Pointing to credit_scores.csv. Ensure your backend saves the heuristic output to this filename or rename appropriately if evaluating new pipeline.
    fetch('/data/credit_scores.csv')
      .then((res) => res.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            setFarmData(results.data);
            if (results.data.length > 0) setActiveFarm(results.data[0]);
          },
        });
      });
  }, []);

  const geoStyle = {
    color: '#10b981',
    weight: 2,
    opacity: 0.6,
    fillColor: '#064e3b',
    fillOpacity: 0.2,
  };

  // Mock trend data for demonstration (in production, extract this from multi-year CSV rows grouped by pointId)
  const trendData = [
    { year: '2019', ndvi: 0.32, rain: 64 },
    { year: '2020', ndvi: 0.39, rain: 72 },
    { year: '2021', ndvi: 0.41, rain: 68 },
    { year: '2022', ndvi: 0.45, rain: 81 },
    { year: '2023', ndvi: activeFarm?.predicted_ndvi || activeFarm?.crop_health_score / 100 || 0.43, rain: 76 },
  ];

  if (!geoData || farmData.length === 0) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center text-emerald-400">
        <Activity className="animate-spin mr-3" size={32} /> Loading Geospatial Intelligence...
      </div>
    );
  }

  const score = activeFarm?.heuristic_credit_score !== undefined ? activeFarm.heuristic_credit_score : activeFarm?.credit_score || 0;
  
  const getRiskColor = (s) => {
    if (s >= 70) return '#10b981'; // Emerald 500
    if (s >= 45) return '#f59e0b'; // Amber 500
    return '#ef4444'; // Red 500
  };

  const getRiskLabel = (s) => {
    if (s >= 70) return 'LOW RISK';
    if (s >= 45) return 'MODERATE RISK';
    return 'HIGH RISK';
  };

  const riskColor = getRiskColor(score);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans p-6 overflow-hidden flex flex-col">
      {/* Header */}
      <header className="flex justify-between items-center mb-6 pb-4 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <Leaf className="text-emerald-500" size={32} />
          <div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-teal-200">
              TerraTrust Intelligence
            </h1>
            <p className="text-slate-400 text-xs tracking-widest uppercase">Academic AgriTech Dashboard</p>
          </div>
        </div>
        <div className="flex bg-slate-800 rounded-lg p-2 border border-slate-700">
          <div className="px-4 text-center border-r border-slate-700">
            <p className="text-xs text-slate-400">Total Scans</p>
            <p className="font-mono text-emerald-400">{farmData.length}</p>
          </div>
          <div className="px-4 text-center">
            <p className="text-xs text-slate-400">Region</p>
            <p className="text-sm font-semibold">Davangere</p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-grow">
        
        {/* Left Panel: Map */}
        <div className="lg:col-span-8 flex flex-col space-y-6">
          <div className="bg-slate-800 rounded-2xl p-1 border border-slate-700 shadow-2xl h-[500px] overflow-hidden relative">
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1000] bg-slate-900/80 backdrop-blur px-4 py-2 rounded-full border border-slate-700 text-xs font-semibold tracking-wider text-emerald-300 pointer-events-none">
              SATELLITE SECTOR OVERVIEW
            </div>
            <MapContainer 
              center={[14.4666, 75.9242]} 
              zoom={9} 
              style={{ height: '100%', width: '100%', borderRadius: '14px' }}
              zoomControl={false}
            >
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                attribution='&copy; <a href="https://carto.com/">CARTO</a>'
              />
              <GeoJSON data={geoData} style={geoStyle} />
              
              {farmData.map((farm, idx) => {
                if (!farm.latitude || !farm.longitude) return null;
                const fScore = farm.heuristic_credit_score !== undefined ? farm.heuristic_credit_score : farm.credit_score;
                const isSelected = activeFarm && (activeFarm.point_id_yr === farm.point_id_yr || activeFarm.farm_id === farm.farm_id);
                return (
                  <CircleMarker
                    key={idx}
                    center={[farm.latitude, farm.longitude]}
                    radius={isSelected ? 10 : 6}
                    pathOptions={{
                      color: getRiskColor(fScore),
                      fillColor: getRiskColor(fScore),
                      fillOpacity: isSelected ? 0.9 : 0.4,
                      weight: isSelected ? 3 : 1
                    }}
                    eventHandlers={{ click: () => setActiveFarm(farm) }}
                  >
                    <Popup className="bg-slate-800 border-none rounded-lg custom-popup">
                      <div className="p-2 min-w-[150px] text-slate-800 font-sans">
                        <p className="font-bold border-b pb-1 mb-2">Taluk: {farm.taluk}</p>
                        <p>Score: <strong>{fScore?.toFixed(1)}</strong></p>
                        <p className="text-xs mt-1">Click to analyze</p>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>
          </div>

          {/* Historical Trends Recharts */}
          <div className="bg-slate-800 rounded-2xl p-5 border border-slate-700 shadow-xl min-h-[250px]">
             <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center">
              <Activity size={16} className="mr-2 text-blue-400" />
              5-Year Physical Environmental Trend (Coordinate: {activeFarm?.point_id_yr || activeFarm?.farm_id || "Select Grid"})
            </h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="year" stroke="#94a3b8" tick={{fontSize: 12}} />
                  <YAxis yAxisId="left" stroke="#10b981" tick={{fontSize: 12}} />
                  <YAxis yAxisId="right" orientation="right" stroke="#3b82f6" tick={{fontSize: 12}} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                  <Line yAxisId="left" type="monotone" dataKey="ndvi" name="NDVI Index" stroke="#10b981" strokeWidth={3} dot={{r: 4, fill: '#10b981'}} />
                  <Line yAxisId="right" type="monotone" dataKey="rain" name="Avg Rainfall (mm)" stroke="#3b82f6" strokeWidth={3} dot={{r: 4, fill: '#3b82f6'}} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Panel: Scoring Gauge & Metrics */}
        <div className="lg:col-span-4 flex flex-col space-y-6">
          
          {/* Hero Gauge Card */}
          <div className="bg-gradient-to-b from-slate-800 to-slate-900 rounded-2xl p-6 border border-slate-700 shadow-2xl relative overflow-hidden flex flex-col items-center">
            
            <div className="absolute top-0 w-full h-1" style={{ backgroundColor: riskColor }}></div>

            <h2 className="text-xs uppercase tracking-widest text-slate-400 mb-2 mt-2">Visulized Risk Score</h2>
            <div className="h-48 w-full flex justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart 
                  cx="50%" cy="50%" 
                  innerRadius="70%" outerRadius="100%" 
                  barSize={15} 
                  data={[{ name: 'Score', value: score, fill: riskColor }]} 
                  startAngle={180} endAngle={0}
                >
                  <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
                  <RadialBar
                    background={{ fill: '#334155' }}
                    dataKey="value"
                    cornerRadius={10}
                  />
                </RadialBarChart>
              </ResponsiveContainer>
              {/* Overlay absolute text in center */}
              <div className="absolute top-[40%] flex flex-col items-center">
                <span className="text-5xl font-extrabold text-white tracking-tighter" style={{ textShadow: `0 0 20px ${riskColor}80` }}>
                  {score?.toFixed(0)}
                </span>
                <span className="text-xs font-bold px-3 py-1 rounded-full mt-2" style={{ backgroundColor: `${riskColor}20`, color: riskColor, border: `1px solid ${riskColor}50` }}>
                  {getRiskLabel(score)}
                </span>
              </div>
            </div>

            <div className="w-full mt-4 bg-slate-950/50 rounded-xl p-4 border border-slate-700/50">
               <p className="text-sm font-medium text-slate-300 flex items-start">
                 {score >= 70 ? <ShieldCheck className="mr-2 text-emerald-500 shrink-0" /> : <AlertTriangle className="mr-2 text-amber-500 shrink-0" />}
                 {activeFarm?.recommendation || "System physical capacity mapping active. Awaiting specific recommendation output."}
               </p>
            </div>
          </div>

          {/* 3-Card Scientific Evidence Grid */}
          <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest pl-2">Scientific Heuristics</h3>
          
          <div className="grid grid-cols-1 gap-4">
            {/* NDVI ML Wrapper */}
             <div className="bg-slate-800/80 rounded-xl p-4 border border-slate-700/80 hover:border-emerald-500/50 transition-colors flex items-center">
                <div className="bg-emerald-900/30 p-3 rounded-lg mr-4 border border-emerald-800/50">
                  <Leaf className="text-emerald-400" size={24} />
                </div>
                <div className="flex-grow">
                  <p className="text-xs text-slate-400">ML Predicted NDVI</p>
                  <p className="text-lg font-bold text-slate-100 mt-0.5">
                    {(activeFarm?.predicted_ndvi || activeFarm?.crop_health_score || 0).toFixed(2)}
                  </p>
                </div>
             </div>

             {/* Soil Suitability */}
             <div className="bg-slate-800/80 rounded-xl p-4 border border-slate-700/80 hover:border-amber-500/50 transition-colors flex items-center">
                <div className="bg-amber-900/30 p-3 rounded-lg mr-4 border border-amber-800/50">
                  <MapPin className="text-amber-400" size={24} />
                </div>
                <div className="flex-grow">
                  <p className="text-xs text-slate-400">ISRIC Soil Suitability</p>
                  <p className="text-lg font-bold text-slate-100 mt-0.5 flex items-end justify-between w-full">
                    <span>{activeFarm?.soil_score?.toFixed(1) || "N/A"} <span className="text-xs font-normal text-slate-500">/ 100</span></span>
                    <span className="text-xs px-2 py-0.5 bg-amber-500/10 text-amber-400 rounded border border-amber-500/20">{activeFarm?.soil_label || "Score"}</span>
                  </p>
                </div>
             </div>

             {/* Water Availability */}
             <div className="bg-slate-800/80 rounded-xl p-4 border border-slate-700/80 hover:border-blue-500/50 transition-colors flex items-center">
                <div className="bg-blue-900/30 p-3 rounded-lg mr-4 border border-blue-800/50">
                  <Droplets className="text-blue-400" size={24} />
                </div>
                <div className="flex-grow">
                  <p className="text-xs text-slate-400">Water / GW Depth</p>
                  <p className="text-lg font-bold text-slate-100 mt-0.5">
                    {activeFarm?.water_score?.toFixed(1) || activeFarm?.groundwater_score?.toFixed(1) || "N/A"} <span className="text-xs font-normal text-slate-500">/ 100</span>
                  </p>
                </div>
             </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default App;
