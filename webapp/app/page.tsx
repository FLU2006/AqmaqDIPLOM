"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { Activity, Clock, Search, ShieldCheck, ShieldAlert, Users, Video, VideoOff, RefreshCw, Download, X, Calendar, Plus, Edit2, Check, UploadCloud, Trash2, Camera } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000';

export default function AqmyaqDashboard() {
  const [incidents, setIncidents] = useState<any[]>([]);
  const [memory, setMemory] = useState<any[]>([]);
  const [serverStatus, setServerStatus] = useState<string>('checking'); 
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  
  // States for open profile modal and edit mode
  const [selectedProfile, setSelectedProfile] = useState<any>(null);
  const [isEditingName, setIsEditingName] = useState<boolean>(false);
  const [editNameValue, setEditNameValue] = useState<string>('');

  // States for add new profile modal
  const [isAddModalOpen, setIsAddModalOpen] = useState<boolean>(false);
  const [newProfileName, setNewProfileName] = useState<string>('');
  const [newProfileImagePreview, setNewProfileImagePreview] = useState<string | null>(null);
  const [newProfileImageFile, setNewProfileImageFile] = useState<File | null>(null);

  // States for Photo Analysis (Forensic)
  const [isAnalyzeModalOpen, setIsAnalyzeModalOpen] = useState<boolean>(false);
  const [analyzeFile, setAnalyzeFile] = useState<File | null>(null);
  const [analyzePreview, setAnalyzePreview] = useState<string | null>(null);
  const [analyzeResult, setAnalyzeResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  // States for report export (CSV by dates)
  const [isExportModalOpen, setIsExportModalOpen] = useState<boolean>(false);
  const [exportStartDate, setExportStartDate] = useState<string>('');
  const [exportEndDate, setExportEndDate] = useState<string>('');

  // Camera state
  const [isCameraEnabled, setIsCameraEnabled] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<boolean>(false);
  const [cameraKey, setCameraKey] = useState<number>(Date.now()); 

  // Lock body scroll when modals are open
  useEffect(() => {
    if (selectedProfile || isAddModalOpen || isAnalyzeModalOpen || isExportModalOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [selectedProfile, isAddModalOpen, isAnalyzeModalOpen, isExportModalOpen]);

  const fetchData = async () => {
    try {
      const healthRes = await fetch(`${API_BASE_URL}/api/health`);
      if (healthRes.ok) setServerStatus('online');
      else setServerStatus('offline');

      const incRes = await fetch(`${API_BASE_URL}/api/incidents?limit=100`);
      if (incRes.ok) {
        const incData = await incRes.json();
        setIncidents(incData);
      }

      const memRes = await fetch(`${API_BASE_URL}/api/memory`);
      if (memRes.ok) {
        const memData = await memRes.json();
        setMemory(memData.profiles || []);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
      setServerStatus('offline');
    }
  };

  useEffect(() => {
    fetchData(); 
    
    let interval: any;
    if (autoRefresh) {
      interval = setInterval(fetchData, 3000); 
    }
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const filteredIncidents = useMemo(() => {
    if (!searchQuery) return incidents;
    return incidents.filter((inc: any) => 
      inc.person_id.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [incidents, searchQuery]);

  const filteredMemory = useMemo(() => {
    if (!searchQuery) return memory;
    return memory.filter((prof: any) => 
      prof.person_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (prof.display_name && prof.display_name.toLowerCase().includes(searchQuery.toLowerCase()))
    );
  }, [memory, searchQuery]);

  const formatTime = (isoString: string) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const formatDate = (isoString: string) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', { day: '2-digit', month: 'short', year: 'numeric' });
  };

  const openExportModal = () => {
    const today = new Date().toISOString().split('T')[0];
    setExportStartDate(today);
    setExportEndDate(today);
    setIsExportModalOpen(true);
  };

  const executeExportToCSV = () => {
    if (incidents.length === 0) return;

    const filteredForExport = incidents.filter(inc => {
      if (!inc.ts) return false;
      const incDate = inc.ts.split('T')[0]; 
      return incDate >= exportStartDate && incDate <= exportEndDate;
    });

    if (filteredForExport.length === 0) {
      alert("No incidents found for the selected period.");
      return;
    }

    const headers = ['Date', 'Time', 'Face ID', 'Display Name', 'Status', 'Accuracy (%)'];
    const rows = filteredForExport.map(inc => {
      const date = new Date(inc.ts);
      const prof = memory.find(p => p.person_id === inc.person_id);
      const displayName = prof?.display_name || inc.person_id;
      const isKnown = !inc.person_id.startsWith('Unknown') || (prof?.display_name && !prof.display_name.startsWith('Unknown'));

      return [
        date.toLocaleDateString('en-US'),
        date.toLocaleTimeString('en-US', { hour12: false }),
        inc.person_id,
        displayName,
        isKnown ? 'Known' : 'Unknown',
        (inc.score * 100).toFixed(1)
      ].join(',');
    });
    
    const csvContent = "data:text/csv;charset=utf-8,\uFEFF" + [headers.join(','), ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `aqmyaq_report_${exportStartDate}_to_${exportEndDate}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setIsExportModalOpen(false);
  };

  const getProfileImage = (profile: any) => {
    const lastIncident = incidents.find((inc: any) => inc.person_id === profile.person_id && inc.image_url);
    if (lastIncident) return lastIncident.image_url;
    
    if (profile.image_urls && profile.image_urls.length > 0) {
      const validUrls = profile.image_urls.filter((url: string) => !url.includes('placeholder') && !url.includes('enroll_async'));
      if (validUrls.length > 0) return validUrls[validUrls.length - 1];
    }
    return null;
  };

  // === FUNCTIONS ===

  const handleSaveName = async () => {
    if (!selectedProfile) return;
    
    const updatedProfile = { ...selectedProfile, display_name: editNameValue };
    setSelectedProfile(updatedProfile);
    setMemory(prev => prev.map(p => p.person_id === selectedProfile.person_id ? updatedProfile : p));
    setIsEditingName(false);

    try {
      await fetch(`${API_BASE_URL}/api/profiles/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_id: selectedProfile.person_id, display_name: editNameValue })
      });
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeleteProfile = async () => {
    if (!selectedProfile) return;
    
    const confirmDelete = window.confirm(`Are you sure you want to permanently delete the profile "${selectedProfile.display_name}"?`);
    if (!confirmDelete) return;

    const pidToRemove = selectedProfile.person_id;

    setMemory(prev => prev.filter(p => p.person_id !== pidToRemove));
    setSelectedProfile(null);

    try {
      await fetch(`${API_BASE_URL}/api/profiles/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_id: pidToRemove })
      });
    } catch (e) {
      console.error('Error deleting profile:', e);
    }
  };

  const handleImageChange = (e: any) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setNewProfileImageFile(file);
      setNewProfileImagePreview(URL.createObjectURL(file));
    }
  };

  const handleAddNewProfile = async () => {
    if (!newProfileName || !newProfileImageFile) return;

    const dummyId = `Known_${newProfileName.replace(/\s+/g, '_')}_${Date.now().toString().slice(-4)}`;
    const newProf = {
      person_id: dummyId,
      display_name: newProfileName,
      samples: 1,
      image_urls: [newProfileImagePreview]
    };
    
    setMemory(prev => [newProf, ...prev]);
    setIsAddModalOpen(false);
    setNewProfileName('');
    setNewProfileImageFile(null);
    setNewProfileImagePreview(null);

    const formData = new FormData();
    formData.append('file', newProfileImageFile);
    formData.append('display_name', newProfileName);

    try {
      await fetch(`${API_BASE_URL}/api/profiles/add`, {
        method: 'POST',
        body: formData
      });
    } catch (e) {
      console.error(e);
    }
  };

  const handleAnalyzeImageChange = (e: any) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAnalyzeFile(file);
      setAnalyzePreview(URL.createObjectURL(file));
      setAnalyzeResult(null);
    }
  };

  const handleAnalyzeSubmit = async () => {
    if (!analyzeFile) return;
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', analyzeFile);

    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze/image`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        setAnalyzeResult(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleCamera = () => {
    if (!isCameraEnabled) {
      setCameraKey(Date.now()); 
      setCameraError(false);
    }
    setIsCameraEnabled(!isCameraEnabled);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans selection:bg-indigo-500/30">
      
      {/* HEADER */}
      <header className="sticky top-0 z-40 bg-gray-900/80 backdrop-blur-md border-b border-gray-800 px-6 py-4 shadow-sm">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <Video className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white">Aqmyaq<span className="text-indigo-400">Recognition</span></h1>
              <p className="text-xs text-gray-400 font-medium">Real-time Face Analytics</p>
            </div>
          </div>

          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2 bg-gray-800 px-3 py-1.5 rounded-full border border-gray-700">
              <div className={`w-2.5 h-2.5 rounded-full ${
                serverStatus === 'online' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 
                serverStatus === 'offline' ? 'bg-rose-500' : 'bg-amber-500 animate-pulse'
              }`}></div>
              <span className="text-sm font-medium">
                {serverStatus === 'online' ? 'System Online' : 
                 serverStatus === 'offline' ? 'Server Offline' : 'Connecting...'}
              </span>
            </div>

            <button 
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-full transition-colors ${
                autoRefresh ? 'bg-indigo-500/10 text-indigo-400' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin-slow' : ''}`} />
              <span className="text-sm font-medium">{autoRefresh ? 'Live Updates' : 'Paused'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* WARNING MESSAGE IF OFFLINE */}
      {serverStatus === 'offline' && (
        <div className="bg-rose-500/10 border border-rose-500/20 m-6 p-4 rounded-xl flex items-center justify-center space-x-3 text-rose-400 max-w-7xl mx-auto">
          <Activity className="w-5 h-5" />
          <p>Failed to connect to the API server. Ensure that <b>uvicorn src.server:app</b> is running on port 8000.</p>
        </div>
      )}

      {/* MAIN CONTENT */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        
        {/* CONTROLS BAR */}
        <div className="flex flex-col sm:flex-row justify-between items-center mb-8 gap-4">
          
          <div className="flex items-center space-x-4 w-full sm:w-auto">
            <div className="relative group w-full sm:w-80">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 group-focus-within:text-indigo-400 transition-colors" />
              <input 
                type="text" 
                placeholder="Search by ID or Name..." 
                value={searchQuery}
                onChange={(e: any) => setSearchQuery(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-xl py-2.5 pl-10 pr-4 text-sm text-gray-200 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all placeholder:text-gray-600"
              />
            </div>
          </div>

          <div className="flex space-x-4 w-full sm:w-auto overflow-x-auto pb-2 sm:pb-0">
            {/* PHOTO ANALYSIS BUTTON */}
            <button 
              onClick={() => setIsAnalyzeModalOpen(true)}
              className="bg-indigo-600 hover:bg-indigo-500 border border-indigo-500 rounded-xl px-4 py-2 flex items-center space-x-2 transition-colors min-w-max text-white"
            >
              <UploadCloud className="w-4 h-4" />
              <span className="text-sm font-semibold">Analyze Photo</span>
            </button>

            <button 
              onClick={openExportModal}
              disabled={incidents.length === 0}
              className="bg-gray-800 hover:bg-indigo-600 border border-gray-700 hover:border-indigo-500 rounded-xl px-4 py-2 flex items-center space-x-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-w-max"
            >
              <Download className="w-4 h-4" />
              <span className="text-sm font-semibold">Download Report</span>
            </button>

            <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-2 flex items-center space-x-3 min-w-max">
              <Activity className="w-4 h-4 text-indigo-400" />
              <div className="flex flex-col">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider font-bold">Incidents</span>
                <span className="text-sm font-semibold">{incidents.length} per session</span>
              </div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-2 flex items-center space-x-3 min-w-max">
              <Users className="w-4 h-4 text-emerald-400" />
              <div className="flex flex-col">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider font-bold">In Database</span>
                <span className="text-sm font-semibold">{memory.length} profiles</span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT PANEL: CAMERA + TIMELINE */}
          <div className="lg:col-span-4 flex flex-col h-[calc(100vh-160px)] gap-6">
            
            {/* LIVE CAMERA FEED */}
            <div className="flex flex-col shrink-0">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-bold flex items-center space-x-2 text-gray-200">
                  <Camera className="w-5 h-5 text-indigo-400" />
                  <span>Live Camera</span>
                </h2>
                
                {/* CAMERA ON/OFF BUTTON */}
                <button
                  onClick={toggleCamera}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border transition-all text-xs font-bold ${
                    isCameraEnabled
                      ? 'bg-rose-500/10 text-rose-400 border-rose-500/20 hover:bg-rose-500/20'
                      : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20'
                  }`}
                >
                  {isCameraEnabled ? <VideoOff className="w-4 h-4" /> : <Video className="w-4 h-4" />}
                  <span>{isCameraEnabled ? 'Disable' : 'Enable'}</span>
                </button>
              </div>
              
              <div className="relative w-full aspect-video bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden shadow-lg group">
                {isCameraEnabled ? (
                  <>
                    <img 
                      src={`${API_BASE_URL}/api/video_feed?k=${cameraKey}`} 
                      alt="Live Camera Feed"
                      className={`w-full h-full object-cover ${cameraError ? 'hidden' : 'block'}`}
                      onError={() => setCameraError(true)}
                      onLoad={() => setCameraError(false)}
                    />
                    {/* Fallback if camera feed is not ready */}
                    {cameraError && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 bg-gray-900">
                        <VideoOff className="w-10 h-10 mb-2 opacity-50" />
                        <span className="text-sm font-medium">Waiting for video stream...</span>
                        <span className="text-xs mt-1 text-center px-4">Start the camera script and refresh the server</span>
                      </div>
                    )}
                    {/* Live Indicator */}
                    {!cameraError && (
                      <div className="absolute top-3 right-3 flex items-center space-x-1.5 bg-black/60 px-2.5 py-1 rounded-lg backdrop-blur-sm border border-gray-700/50">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                        <span className="text-[10px] font-bold text-white uppercase tracking-wider">Live</span>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 bg-gray-900/50">
                    <Camera className="w-10 h-10 mb-3 opacity-30" />
                    <span className="text-sm font-medium">Camera is disabled</span>
                    <span className="text-xs mt-1 text-gray-500">Click the "Enable" button</span>
                  </div>
                )}
              </div>
            </div>

            {/* TIMELINE */}
            <div className="flex flex-col flex-1 min-h-0">
              <h2 className="text-lg font-bold mb-3 flex items-center space-x-2 text-gray-200 shrink-0">
                <Clock className="w-5 h-5 text-indigo-400" />
                <span>Event Timeline</span>
              </h2>
              
              <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
                {filteredIncidents.length === 0 ? (
                  <div className="text-center py-10 text-gray-500 border border-dashed border-gray-800 rounded-xl">
                    {searchQuery ? 'Nothing found' : 'Waiting for events...'}
                  </div>
                ) : (
                  filteredIncidents.map((incident: any, idx: number) => {
                    const prof = memory.find(p => p.person_id === incident.person_id);
                    const displayName = prof?.display_name || incident.person_id;
                    const isKnown = !incident.person_id.startsWith('Unknown') || (prof?.display_name && !prof.display_name.startsWith('Unknown'));

                    return (
                      <div key={idx} className="group bg-gray-900 border border-gray-800 hover:border-indigo-500/50 rounded-xl p-3 flex space-x-4 transition-all hover:shadow-[0_4px_20px_-10px_rgba(99,102,241,0.2)]">
                        {/* Thumbnail */}
                        <div className="relative w-16 h-16 rounded-lg overflow-hidden shrink-0 bg-gray-800 border border-gray-700">
                          {incident.image_url ? (
                            <img 
                              src={incident.image_url.startsWith('blob:') ? incident.image_url : `${API_BASE_URL}${incident.image_url}`} 
                              alt="Face crop" 
                              className="w-full h-full object-cover object-center"
                              onError={(e: any) => { e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHg9IjMiIHk9IjMiIHJ4PSIyIiByeT0iMiIvPjxjaXJjbGUgY3g9IjkiIGN5PSI5IiByPSIyIi8+PHBhdGggZD0ibTIxIDE1LTMuMDgtMy4wOGExLjIgMS4yIDAgMDAtMS42Ni4wMGwtNS4wNiA1LjA2Ii8+PC9zdmc+'; }}
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-gray-600"><Users className="w-6 h-6"/></div>
                          )}
                          <div className={`absolute bottom-0 inset-x-0 h-1 ${isKnown ? 'bg-emerald-500' : 'bg-amber-500'}`}></div>
                        </div>

                        {/* Details */}
                        <div className="flex-1 min-w-0 py-0.5">
                          <div className="flex justify-between items-start mb-1">
                            <p className="text-sm font-bold text-gray-100 truncate pr-2">
                              {displayName}
                            </p>
                            <span className="text-xs text-gray-500 whitespace-nowrap">{formatTime(incident.ts)}</span>
                          </div>
                          
                          <div className="flex items-center space-x-2 mt-2">
                            {isKnown ? (
                              <div className="flex items-center text-[10px] uppercase font-bold tracking-wider text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
                                <ShieldCheck className="w-3 h-3 mr-1" /> Known
                              </div>
                            ) : (
                              <div className="flex items-center text-[10px] uppercase font-bold tracking-wider text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20">
                                <ShieldAlert className="w-3 h-3 mr-1" /> Unknown
                              </div>
                            )}
                            <span className="text-xs text-gray-400 font-mono bg-gray-800 px-1.5 py-0.5 rounded border border-gray-700">
                              {(incident.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* RIGHT PANEL: GALLERY */}
          <div className="lg:col-span-8 flex flex-col h-[calc(100vh-160px)]">
            <div className="flex items-center justify-between mb-4 shrink-0">
              <h2 className="text-lg font-bold flex items-center space-x-2 text-gray-200">
                <Users className="w-5 h-5 text-emerald-400" />
                <span>Face Database (Memory)</span>
              </h2>
              {/* ADD NEW PERSON BUTTON */}
              <button 
                onClick={() => setIsAddModalOpen(true)}
                className="flex items-center space-x-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 px-3 py-1.5 rounded-lg transition-colors text-sm font-bold"
              >
                <Plus className="w-4 h-4" />
                <span>Add</span>
              </button>
            </div>

            <div className="flex-1 bg-gray-900/50 border border-gray-800 rounded-2xl p-6 overflow-y-auto custom-scrollbar">
              {filteredMemory.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-4">
                  <Users className="w-12 h-12 opacity-20" />
                  <p>{searchQuery ? 'No profiles found' : 'Database is currently empty'}</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-4 gap-4">
                  {filteredMemory.map((profile: any, idx: number) => {
                    const isKnown = !profile.person_id.startsWith('Unknown') || (profile.display_name && !profile.display_name.startsWith('Unknown'));
                    const latestImage = getProfileImage(profile);

                    return (
                      <div 
                        key={idx} 
                        onClick={() => {
                          setSelectedProfile(profile);
                          setEditNameValue(profile.display_name);
                          setIsEditingName(false);
                        }}
                        className="group relative bg-gray-800 border border-gray-700 rounded-xl overflow-hidden hover:border-indigo-500 transition-colors cursor-pointer"
                      >
                        {/* Status Ribbon */}
                        <div className={`absolute top-0 inset-x-0 h-1 z-10 ${isKnown ? 'bg-emerald-500' : 'bg-amber-500'}`}></div>
                        
                        <div className="aspect-square w-full bg-gray-900 relative overflow-hidden">
                          {latestImage ? (
                            <img 
                              src={latestImage.startsWith('blob:') ? latestImage : `${API_BASE_URL}${latestImage}`} 
                              alt={profile.display_name} 
                              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                              onError={(e: any) => { e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YjU1NjMiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHg9IjMiIHk9IjMiIHJ4PSIyIiByeT0iMiIvPjxjaXJjbGUgY3g9IjkiIGN5PSI5IiByPSIyIi8+PHBhdGggZD0ibTIxIDE1LTMuMDgtMy4wOGExLjIgMS4yIDAgMDAtMS42Ni4wMGwtNS4wNiA1LjA2Ii8+PC9zdmc+'; }}
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-gray-700">No Image</div>
                          )}
                          
                          {/* Samples Counter Badge */}
                          <div className="absolute top-2 right-2 bg-gray-900/80 backdrop-blur text-gray-300 text-[10px] font-bold px-2 py-1 rounded border border-gray-700 shadow-sm">
                            {profile.samples} samples
                          </div>
                        </div>

                        <div className="p-3 bg-gray-800 group-hover:bg-gray-800/80 transition-colors">
                          <h3 className="text-sm font-bold text-white truncate" title={profile.display_name}>
                            {profile.display_name}
                          </h3>
                          <p className="text-[10px] text-gray-500 mt-0.5 truncate font-mono">{profile.person_id}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

        </div>
      </main>

      {/* =========================================================
          MODAL: PROFILE INFORMATION & EDIT
          ========================================================= */}
      {selectedProfile && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/80 backdrop-blur-sm animate-in fade-in duration-200"
          onClick={() => setSelectedProfile(null)}
        >
          <div 
            className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-2xl flex flex-col max-h-[90vh] relative overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            
            <button 
              onClick={() => setSelectedProfile(null)}
              className="absolute top-4 right-4 p-2 bg-gray-900/50 hover:bg-rose-500/80 text-gray-300 hover:text-white rounded-full transition-colors z-50 cursor-pointer backdrop-blur-md"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex-1 overflow-y-auto custom-scrollbar">
              <div className="h-32 bg-gradient-to-r from-indigo-900/40 to-gray-900 border-b border-gray-700 w-full shrink-0"></div>

              <div className="px-6 pb-6 relative">
                
                <div className="flex flex-col sm:flex-row gap-6 -mt-16 mb-8 items-end sm:items-center relative z-10">
                  <div className="w-32 h-32 rounded-2xl border-4 border-gray-900 overflow-hidden bg-gray-800 shadow-xl shrink-0">
                    {getProfileImage(selectedProfile) ? (
                      <img 
                        src={getProfileImage(selectedProfile).startsWith('blob:') ? getProfileImage(selectedProfile) : `${API_BASE_URL}${getProfileImage(selectedProfile)}`} 
                        alt="Avatar" 
                        className="w-full h-full object-cover bg-gray-800"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-gray-800"><Users className="w-10 h-10 text-gray-600"/></div>
                    )}
                  </div>
                  <div className="flex-1 pb-2 w-full flex items-center justify-between">
                    <div>
                      <div className="flex items-center space-x-3 mb-1">
                        
                        {/* EDIT NAME */}
                        {isEditingName ? (
                          <div className="flex items-center space-x-2 w-full max-w-xs">
                            <input 
                              type="text" 
                              value={editNameValue}
                              onChange={(e) => setEditNameValue(e.target.value)}
                              className="bg-gray-800 border border-indigo-500 text-white px-3 py-1.5 rounded-lg text-lg font-bold w-full focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                              autoFocus
                              onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
                            />
                            <button onClick={handleSaveName} className="p-1.5 bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg transition-colors">
                              <Check className="w-5 h-5" />
                            </button>
                          </div>
                        ) : (
                          <>
                            <h2 className="text-2xl font-bold text-white truncate max-w-[200px] sm:max-w-xs">{selectedProfile.display_name}</h2>
                            <button onClick={() => setIsEditingName(true)} className="p-1 text-gray-500 hover:text-indigo-400 transition-colors">
                              <Edit2 className="w-4 h-4" />
                            </button>
                          </>
                        )}

                        {(!selectedProfile.person_id.startsWith('Unknown') || (selectedProfile.display_name && !selectedProfile.display_name.startsWith('Unknown'))) ? (
                          <span className="bg-emerald-500/20 text-emerald-400 text-xs px-2 py-1 rounded font-bold uppercase tracking-wide border border-emerald-500/20 shrink-0">Known</span>
                        ) : (
                          <span className="bg-amber-500/20 text-amber-400 text-xs px-2 py-1 rounded font-bold uppercase tracking-wide border border-amber-500/20 shrink-0">Unknown</span>
                        )}
                      </div>
                      <p className="text-sm text-gray-400 font-mono">ID: {selectedProfile.person_id}</p>
                    </div>

                    {/* DELETE PROFILE BUTTON */}
                    <button 
                      onClick={handleDeleteProfile} 
                      className="p-2 bg-gray-800/50 hover:bg-rose-500/20 text-gray-500 hover:text-rose-400 border border-gray-700/50 hover:border-rose-500/50 rounded-xl transition-all ml-4 shrink-0"
                      title="Permanently delete profile"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-8">
                  <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                    <p className="text-xs text-gray-500 uppercase font-bold mb-1">Samples in Database</p>
                    <p className="text-xl font-semibold text-gray-200">{selectedProfile.samples}</p>
                  </div>
                  <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                    <p className="text-xs text-gray-500 uppercase font-bold mb-1">Total Visits (Session)</p>
                    <p className="text-xl font-semibold text-gray-200">
                      {incidents.filter((i: any) => i.person_id === selectedProfile.person_id).length}
                    </p>
                  </div>
                </div>

                <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-4 flex items-center">
                  <Calendar className="w-4 h-4 mr-2 text-indigo-400" /> 
                  Appearance History
                </h3>
                
                <div className="space-y-3">
                  {incidents.filter((i: any) => i.person_id === selectedProfile.person_id).length === 0 ? (
                    <p className="text-sm text-gray-500 italic">No incidents found in the current session.</p>
                  ) : (
                    incidents
                      .filter((i: any) => i.person_id === selectedProfile.person_id)
                      .map((inc: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between bg-gray-800/30 p-3 rounded-lg border border-gray-700/50">
                        <div className="flex items-center space-x-4">
                          <div className="w-10 h-10 rounded bg-gray-800 overflow-hidden shrink-0">
                            {inc.image_url && <img src={inc.image_url.startsWith('blob:') ? inc.image_url : `${API_BASE_URL}${inc.image_url}`} className="w-full h-full object-cover" alt="crop"/>}
                          </div>
                          <div>
                            <p className="text-sm text-gray-200">{formatDate(inc.ts)}</p>
                            <p className="text-xs text-gray-500">{formatTime(inc.ts)}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-gray-500 uppercase">Accuracy</p>
                          <p className="text-sm font-mono text-indigo-400">{(inc.score * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    ))
                  )}
                </div>

              </div>
            </div>
          </div>
        </div>
      )}

      {/* =========================================================
          MODAL: ADD NEW PERSON WITH PHOTO
          ========================================================= */}
      {isAddModalOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/80 backdrop-blur-sm animate-in fade-in duration-200"
          onClick={() => setIsAddModalOpen(false)}
        >
          <div 
            className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-md flex flex-col relative overflow-hidden p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-white flex items-center">
                <Users className="w-5 h-5 mr-2 text-emerald-400" /> Add Person
              </h2>
              <button onClick={() => setIsAddModalOpen(false)} className="text-gray-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-5">
              {/* Photo Upload */}
              <div>
                <label className="block text-sm font-bold text-gray-400 mb-2 uppercase tracking-wide">Face Photo</label>
                <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-700 hover:border-indigo-500 hover:bg-indigo-500/5 rounded-xl cursor-pointer transition-all overflow-hidden group">
                  {newProfileImagePreview ? (
                    <img src={newProfileImagePreview} alt="Preview" className="w-full h-full object-contain" />
                  ) : (
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <UploadCloud className="w-10 h-10 text-gray-500 group-hover:text-indigo-400 mb-3 transition-colors" />
                      <p className="mb-2 text-sm text-gray-400"><span className="font-semibold text-indigo-400">Click</span> or drag & drop photo</p>
                      <p className="text-xs text-gray-500">Face only (PNG, JPG)</p>
                    </div>
                  )}
                  <input type="file" className="hidden" accept="image/*" onChange={handleImageChange} />
                </label>
              </div>

              {/* Name */}
              <div>
                <label className="block text-sm font-bold text-gray-400 mb-2 uppercase tracking-wide">Full Name</label>
                <input 
                  type="text" 
                  value={newProfileName}
                  onChange={(e) => setNewProfileName(e.target.value)}
                  placeholder="E.g., John Doe"
                  className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                />
              </div>

              {/* Save Button */}
              <button 
                onClick={handleAddNewProfile}
                disabled={!newProfileName || !newProfileImageFile}
                className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mt-4"
              >
                Save to Database
              </button>
            </div>
            
          </div>
        </div>
      )}

      {/* =========================================================
          MODAL: PHOTO ANALYSIS (FORENSIC)
          ========================================================= */}
      {isAnalyzeModalOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/80 backdrop-blur-sm animate-in fade-in duration-200"
          onClick={() => setIsAnalyzeModalOpen(false)}
        >
          <div 
            className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-4xl flex flex-col relative overflow-hidden p-6 max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-6 shrink-0">
              <h2 className="text-xl font-bold text-white flex items-center">
                <Search className="w-5 h-5 mr-2 text-indigo-400" /> Group Photo Analysis
              </h2>
              <button onClick={() => setIsAnalyzeModalOpen(false)} className="text-gray-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
              {/* UPLOAD SCREEN */}
              {!analyzeResult && (
                <div>
                  <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-gray-700 hover:border-indigo-500 hover:bg-indigo-500/5 rounded-xl cursor-pointer transition-all overflow-hidden group">
                    {analyzePreview ? (
                      <img src={analyzePreview} alt="Preview" className="w-full h-full object-contain" />
                    ) : (
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <UploadCloud className="w-12 h-12 text-gray-500 group-hover:text-indigo-400 mb-3 transition-colors" />
                        <p className="mb-2 text-sm text-gray-400"><span className="font-semibold text-indigo-400">Click</span> or drag & drop group photo</p>
                        <p className="text-xs text-gray-500">JPEG, PNG (detect all faces)</p>
                      </div>
                    )}
                    <input type="file" className="hidden" accept="image/*" onChange={handleAnalyzeImageChange} />
                  </label>

                  <button 
                    onClick={handleAnalyzeSubmit}
                    disabled={!analyzeFile || isAnalyzing}
                    className="w-full py-3 mt-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    {isAnalyzing ? <RefreshCw className="w-5 h-5 animate-spin" /> : 'Start Analysis'}
                  </button>
                </div>
              )}

              {/* ANALYSIS RESULT SCREEN */}
              {analyzeResult && (
                <div className="flex flex-col gap-4">
                  <div className="w-full bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
                    <img 
                      src={`${API_BASE_URL}${analyzeResult.image_url}`} 
                      alt="Analyzed" 
                      className="w-full h-auto object-contain max-h-[50vh]"
                    />
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-bold text-white mb-3">Faces found: {analyzeResult.faces_found}</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {analyzeResult.results.map((res: any, idx: number) => (
                        <div key={idx} className="bg-gray-800 p-3 rounded-lg border border-gray-700 flex justify-between items-center">
                          <div>
                            <p className="font-bold text-gray-200">{res.display_name}</p>
                            <p className="text-xs text-gray-500 font-mono">{res.person_id || 'Unknown'}</p>
                          </div>
                          {res.status === 'known' ? (
                            <span className="text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded text-xs font-bold">KNOWN {(res.score*100).toFixed(0)}%</span>
                          ) : (
                            <span className="text-amber-400 bg-amber-400/10 px-2 py-1 rounded text-xs font-bold">UNKNOWN</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <button 
                    onClick={() => { setAnalyzeResult(null); setAnalyzePreview(null); setAnalyzeFile(null); }}
                    className="w-full py-3 mt-2 bg-gray-800 hover:bg-gray-700 text-white font-bold rounded-xl transition-colors border border-gray-700"
                  >
                    Upload another photo
                  </button>
                </div>
              )}
            </div>
            
          </div>
        </div>
      )}

      {/* =========================================================
          MODAL: EXPORT CSV REPORT (DATE FILTER)
          ========================================================= */}
      {isExportModalOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/80 backdrop-blur-sm animate-in fade-in duration-200"
          onClick={() => setIsExportModalOpen(false)}
        >
          <div 
            className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-sm flex flex-col relative overflow-hidden p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-white flex items-center">
                <Download className="w-5 h-5 mr-2 text-indigo-400" /> Export Report
              </h2>
              <button onClick={() => setIsExportModalOpen(false)} className="text-gray-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <p className="text-sm text-gray-400">Select a period to export incidents to CSV.</p>
              
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="block text-xs font-bold text-gray-500 mb-1 uppercase tracking-wide">Start Date</label>
                  <input 
                    type="date" 
                    value={exportStartDate}
                    onChange={(e) => setExportStartDate(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 [color-scheme:dark]"
                  />
                </div>
                <div className="flex-1">
                  <label className="block text-xs font-bold text-gray-500 mb-1 uppercase tracking-wide">End Date</label>
                  <input 
                    type="date" 
                    value={exportEndDate}
                    onChange={(e) => setExportEndDate(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 [color-scheme:dark]"
                  />
                </div>
              </div>

              <button 
                onClick={executeExportToCSV}
                disabled={!exportStartDate || !exportEndDate || exportStartDate > exportEndDate}
                className="w-full py-3 mt-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                <Download className="w-4 h-4 mr-2" /> Generate CSV
              </button>
            </div>
          </div>
        </div>
      )}

      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #374151; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #4B5563; }
        .animate-spin-slow { animation: spin 3s linear infinite; }
      `}} />
    </div>
  );
}