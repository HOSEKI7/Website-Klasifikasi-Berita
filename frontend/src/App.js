import { useState, useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Toaster, toast } from "sonner";
import { Newspaper, TrendingUp, Sparkles, BarChart3 } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Home = () => {
  const [judul, setJudul] = useState("");
  const [isi, setIsi] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastClassified, setLastClassified] = useState(null);
  const [allNews, setAllNews] = useState([]);
  const [stats, setStats] = useState([]);
  const [activeTab, setActiveTab] = useState("input");

  const categories = ["Politik", "Olahraga", "Teknologi", "Hiburan", "Ekonomi"];

  const categoryColors = {
    "Politik": "bg-blue-500",
    "Olahraga": "bg-green-500",
    "Teknologi": "bg-purple-500",
    "Hiburan": "bg-pink-500",
    "Ekonomi": "bg-amber-500"
  };

  useEffect(() => {
    fetchAllNews();
    fetchStats();
  }, []);

  const fetchAllNews = async () => {
    try {
      const response = await axios.get(`${API}/news`);
      setAllNews(response.data);
    } catch (error) {
      console.error("Error fetching news:", error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/categories/stats`);
      setStats(response.data);
    } catch (error) {
      console.error("Error fetching stats:", error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!judul.trim() || !isi.trim()) {
      toast.error("Judul dan isi berita harus diisi!");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/news`, {
        judul: judul,
        isi: isi
      });
      
      setLastClassified(response.data);
      toast.success(`Berita berhasil diklasifikasikan sebagai ${response.data.kategori}!`);
      
      // Reset form
      setJudul("");
      setIsi("");
      
      // Refresh data
      fetchAllNews();
      fetchStats();
      
      // Switch to results tab
      setActiveTab("all");
    } catch (error) {
      console.error("Error classifying news:", error);
      toast.error("Gagal mengklasifikasikan berita. Silakan coba lagi.");
    } finally {
      setLoading(false);
    }
  };

  const getNewsByCategory = (category) => {
    return allNews.filter(news => news.kategori === category);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('id-ID', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <Toaster position="top-center" richColors />
      
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl">
              <Newspaper className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Klasifikasi Berita</h1>
              <p className="text-sm text-slate-600">Sistem Otomatis dengan TF-IDF & KNN</p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
          {stats.map((stat) => (
            <Card key={stat.kategori} className="border-none shadow-lg hover:shadow-xl transition-shadow duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <span className={`w-3 h-3 rounded-full ${categoryColors[stat.kategori]}`}></span>
                  <BarChart3 className="w-4 h-4 text-slate-400" />
                </div>
                <div className="text-3xl font-bold text-slate-900 mb-1">{stat.jumlah}</div>
                <div className="text-sm text-slate-600">{stat.kategori}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-2 bg-white shadow-md p-1 h-auto">
            <TabsTrigger 
              value="input" 
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-indigo-600 data-[state=active]:text-white py-3"
              data-testid="tab-input"
            >
              <Sparkles className="w-4 h-4 mr-2" />
              Tambah Berita
            </TabsTrigger>
            <TabsTrigger 
              value="all" 
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-indigo-600 data-[state=active]:text-white py-3"
              data-testid="tab-all-news"
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              Daftar Berita
            </TabsTrigger>
          </TabsList>

          <TabsContent value="input" className="space-y-6">
            <Card className="border-none shadow-xl">
              <CardHeader>
                <CardTitle className="text-2xl">Input Berita Baru</CardTitle>
                <CardDescription>Masukkan judul dan isi berita untuk diklasifikasikan secara otomatis</CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700">Judul Berita</label>
                    <Input
                      placeholder="Masukkan judul berita..."
                      value={judul}
                      onChange={(e) => setJudul(e.target.value)}
                      className="h-12 text-base"
                      data-testid="input-judul"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700">Isi Berita</label>
                    <Textarea
                      placeholder="Masukkan isi berita..."
                      value={isi}
                      onChange={(e) => setIsi(e.target.value)}
                      className="min-h-[200px] text-base"
                      data-testid="textarea-isi"
                    />
                  </div>

                  <Button 
                    type="submit" 
                    className="w-full h-12 text-base bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700"
                    disabled={loading}
                    data-testid="button-submit"
                  >
                    {loading ? "Mengklasifikasi..." : "Klasifikasi Berita"}
                  </Button>
                </form>

                {lastClassified && (
                  <div className="mt-6 p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border border-green-200" data-testid="classification-result">
                    <h3 className="font-semibold text-lg mb-3 text-green-900">Hasil Klasifikasi</h3>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-600">Kategori:</span>
                        <Badge className={`${categoryColors[lastClassified.kategori]} text-white`} data-testid="result-kategori">
                          {lastClassified.kategori}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-600">Confidence Score:</span>
                        <span className="font-bold text-green-900" data-testid="result-confidence">
                          {(lastClassified.confidence_score * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="all" className="space-y-6">
            {categories.map((category) => {
              const categoryNews = getNewsByCategory(category);
              return (
                <Card key={category} className="border-none shadow-xl overflow-hidden">
                  <CardHeader className={`${categoryColors[category]} text-white`}>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl">{category}</CardTitle>
                      <Badge variant="secondary" className="bg-white/20 text-white" data-testid={`category-count-${category.toLowerCase()}`}>
                        {categoryNews.length} berita
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-6">
                    {categoryNews.length === 0 ? (
                      <p className="text-slate-500 text-center py-8">Belum ada berita dalam kategori ini</p>
                    ) : (
                      <div className="space-y-4">
                        {categoryNews.map((news) => (
                          <div 
                            key={news.id} 
                            className="p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors duration-200 border border-slate-200"
                            data-testid={`news-item-${news.id}`}
                          >
                            <h4 className="font-semibold text-slate-900 mb-2" data-testid={`news-title-${news.id}`}>{news.judul}</h4>
                            <p className="text-sm text-slate-600 mb-3 line-clamp-2" data-testid={`news-content-${news.id}`}>{news.isi}</p>
                            <div className="flex items-center justify-between text-xs text-slate-500">
                              <span>{formatDate(news.created_at)}</span>
                              <span className="font-medium" data-testid={`news-confidence-${news.id}`}>
                                Confidence: {(news.confidence_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;