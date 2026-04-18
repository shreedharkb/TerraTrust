import { useState, useEffect } from 'react'
import Papa from 'papaparse'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts'
import { Search, Map as MapIcon, ShieldCheck, ShieldAlert, Cpu, Droplet, Sprout, TrendingUp } from 'lucide-react'
import 'leaflet/dist/leaflet.css'

function App() {
  const [farms, setFarms] = useState([])
  const [selectedFarm, setSelectedFarm] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load the farm scores data
    Papa.parse('/data/credit_scores.csv', {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        setFarms(results.data.filter(f => f.farm_id))
        setLoading(false)
      }
    })
  }, [])

  const filteredFarms = farms.filter(f => 
    f.farm_id?.toLowerCase().includes(searchTerm.toLowerCase()) || 
    f.taluk?.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="app-container">
      <header>
        <div className="logo-area">
          <h1><Sprout size={28} color="#10b981" /> TerraTrust Dashboard</h1>
        </div>
        <div style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
          <div style={{background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', padding: '0.5rem 1rem', borderRadius: '20px', fontSize: '0.875rem', fontWeight: 600}}>
            Davangere District
          </div>
        </div>
      </header>

      <div className="main-content">
        <div className="sidebar">
          <div className="sidebar-header">
            <div style={{position: 'relative'}}>
              <Search size={18} style={{position: 'absolute', left: '1rem', top: '10px', color: '#94a3b8'}} />
              <input 
                type="text" 
                className="search-bar" 
                placeholder="Search farm ID or taluk..." 
                style={{paddingLeft: '2.5rem'}}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div style={{marginTop: '1rem', fontSize: '0.8rem', color: '#94a3b8', display: 'flex', justifyContent: 'space-between'}}>
              <span>{filteredFarms.length} Farms</span>
              <span>Sorted by Risk</span>
            </div>
          </div>
          
          <div className="farm-list">
            {loading ? <div className="loader"></div> : 
             filteredFarms.sort((a,b) => a.credit_score - b.credit_score).map(farm => (
              <div 
                key={farm.farm_id} 
                className={`farm-card ${selectedFarm?.farm_id === farm.farm_id ? 'active' : ''}`}
                onClick={() => setSelectedFarm(farm)}
              >
                <div className="farm-header">
                  <span className="farm-title">{farm.farm_id}</span>
                  <span className={`risk-badge ${
                    farm.credit_score >= 75 ? 'risk-low' : 
                    farm.credit_score >= 50 ? 'risk-medium' : 'risk-high'
                  }`}>
                    {Math.round(farm.credit_score)}/100
                  </span>
                </div>
                <div style={{fontSize: '0.8rem', color: '#94a3b8', display: 'flex', gap: '1rem'}}>
                  <span>{farm.taluk}</span>
                  <span>{farm.declared_crop}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="dashboard">
          {!selectedFarm ? (
            <div className="empty-state">
              <MapIcon />
              <h2>Select a farm to view intelligence</h2>
              <p>Choose a farm from the list to see detailed credit scoring</p>
            </div>
          ) : (
            <>
              <div className="dashboard-grid">
                
                {/* Main Score Panel */}
                <div className="glass-panel" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center'}}>
                  <h3 className="panel-title" style={{alignSelf: 'flex-start'}}><ShieldCheck size={20} color="#3b82f6" /> Visual Credit Score</h3>
                  
                  <div className="score-circle" style={{'--score': selectedFarm.credit_score}}>
                    <span className="score-value" style={{
                      color: selectedFarm.credit_score >= 75 ? '#34d399' : 
                             selectedFarm.credit_score >= 50 ? '#fbbf24' : '#f87171'
                    }}>
                      {Math.round(selectedFarm.credit_score)}
                    </span>
                    <span className="score-label">out of 100</span>
                  </div>
                  
                  <h4 style={{fontSize: '1.25rem', marginBottom: '0.5rem',
                    color: selectedFarm.credit_score >= 75 ? '#34d399' : 
                           selectedFarm.credit_score >= 50 ? '#fbbf24' : '#f87171'
                  }}>
                    {selectedFarm.risk_category}
                  </h4>
                  <p style={{fontSize: '0.875rem', color: '#94a3b8', padding: '0 1rem'}}>
                    {selectedFarm.recommendation}
                  </p>
                </div>

                {/* Map Panel */}
                <div className="glass-panel col-span-2">
                  <h3 className="panel-title"><MapIcon size={20} color="#10b981" /> Geospatial Validations</h3>
                  <div className="map-container">
                    <MapContainer 
                      center={[selectedFarm.latitude || 14.4, selectedFarm.longitude || 75.9]} 
                      zoom={11} 
                      style={{height: '100%', width: '100%', background: '#0f172a'}}
                    >
                      <TileLayer
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
                      />
                      <CircleMarker 
                        center={[selectedFarm.latitude || 14.4, selectedFarm.longitude || 75.9]}
                        radius={12}
                        pathOptions={{ 
                          color: selectedFarm.credit_score >= 75 ? '#10b981' : selectedFarm.credit_score >= 50 ? '#f59e0b' : '#ef4444',
                          fillColor: selectedFarm.credit_score >= 75 ? '#10b981' : selectedFarm.credit_score >= 50 ? '#f59e0b' : '#ef4444',
                          fillOpacity: 0.6
                        }}
                      >
                        <Popup>
                          <strong>{selectedFarm.farm_id}</strong><br/>
                          Crop: {selectedFarm.declared_crop}
                        </Popup>
                      </CircleMarker>
                      <MapUpdater lat={selectedFarm.latitude} lng={selectedFarm.longitude} />
                    </MapContainer>
                  </div>
                </div>

                {/* Score Components */}
                <div className="glass-panel col-span-3">
                  <h3 className="panel-title"><Cpu size={20} color="#a855f7" /> ML Engine Factors</h3>
                  <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem'}}>
                    
                    <div style={{background: 'rgba(15, 23, 42, 0.4)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)'}}>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '1rem'}}>
                        <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#a855f7'}}>
                          <Sprout size={18} /> <span style={{fontWeight: 600}}>Crop Health</span>
                        </div>
                        <span style={{fontWeight: 700, fontSize: '1.2rem'}}>{selectedFarm.crop_health_score}/100</span>
                      </div>
                      <div style={{fontSize: '0.875rem', color: '#94a3b8'}}>Classification: <strong>{selectedFarm.crop_health_label}</strong></div>
                      <div style={{width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', marginTop: '1rem', borderRadius: '2px'}}>
                        <div style={{width: `${selectedFarm.crop_health_score}%`, height: '100%', background: '#a855f7', borderRadius: '2px'}}></div>
                      </div>
                    </div>

                    <div style={{background: 'rgba(15, 23, 42, 0.4)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)'}}>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '1rem'}}>
                        <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#3b82f6'}}>
                          <Droplet size={18} /> <span style={{fontWeight: 600}}>Water Avail</span>
                        </div>
                        <span style={{fontWeight: 700, fontSize: '1.2rem'}}>{selectedFarm.water_score}/100</span>
                      </div>
                      <div style={{fontSize: '0.875rem', color: '#94a3b8'}}>Regression: <strong>{selectedFarm.water_label}</strong></div>
                      <div style={{width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', marginTop: '1rem', borderRadius: '2px'}}>
                        <div style={{width: `${selectedFarm.water_score}%`, height: '100%', background: '#3b82f6', borderRadius: '2px'}}></div>
                      </div>
                    </div>

                    <div style={{background: 'rgba(15, 23, 42, 0.4)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)'}}>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '1rem'}}>
                        <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#f59e0b'}}>
                          <MapIcon size={18} /> <span style={{fontWeight: 600}}>Soil Suitability</span>
                        </div>
                        <span style={{fontWeight: 700, fontSize: '1.2rem'}}>{selectedFarm.soil_score}/100</span>
                      </div>
                      <div style={{fontSize: '0.875rem', color: '#94a3b8'}}>Suitability: <strong>{selectedFarm.soil_label}</strong></div>
                      <div style={{width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', marginTop: '1rem', borderRadius: '2px'}}>
                        <div style={{width: `${selectedFarm.soil_score}%`, height: '100%', background: '#f59e0b', borderRadius: '2px'}}></div>
                      </div>
                    </div>

                    <div style={{background: 'rgba(15, 23, 42, 0.4)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)'}}>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '1rem'}}>
                        <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10b981'}}>
                          <TrendingUp size={18} /> <span style={{fontWeight: 600}}>History/Trend</span>
                        </div>
                        <span style={{fontWeight: 700, fontSize: '1.2rem'}}>{selectedFarm.trend_score}/100</span>
                      </div>
                      <div style={{fontSize: '0.875rem', color: '#94a3b8'}}>Stability Score</div>
                      <div style={{width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', marginTop: '1rem', borderRadius: '2px'}}>
                        <div style={{width: `${selectedFarm.trend_score}%`, height: '100%', background: '#10b981', borderRadius: '2px'}}></div>
                      </div>
                    </div>

                  </div>
                </div>

              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function MapUpdater({ lat, lng }) {
  const map = useMap();
  useEffect(() => {
    if (lat && lng) map.setView([lat, lng], 13);
  }, [lat, lng, map]);
  return null;
}

export default App
