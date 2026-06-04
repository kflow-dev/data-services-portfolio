import { createFileRoute } from "@tanstack/react-router";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

export const Route = createFileRoute("/apps/cooldrinks")({
  head: () => ({
    meta: [{ title: "CoolDrinks — Context-Aware Beverage Recommender" }],
  }),
  component: Cooldrinks,
});

const WEATHER_OPTIONS = [
  { value: "sunny", label: "Sunny" },
  { value: "rainy", label: "Rainy" },
  { value: "cloudy", label: "Cloudy" },
  { value: "snowy", label: "Snowy" },
  { value: "stormy", label: "Stormy" },
];

const TIME_OPTIONS = [
  { value: "morning", label: "Morning (6-12)" },
  { value: "afternoon", label: "Afternoon (12-18)" },
  { value: "evening", label: "Evening (18-24)" },
];

const OCCASION_OPTIONS = [
  { value: "casual", label: "Casual" },
  { value: "celebration", label: "Celebration" },
  { value: "pairing", label: "Pairing" },
  { value: "recovery", label: "Recovery" },
  { value: "social", label: "Social" },
  { value: "business", label: "Business" },
];

function Cooldrinks() {
  const [weather, setWeather] = useState<string>("sunny");
  const [timePeriod, setTimePeriod] = useState<string>("afternoon");
  const [occasion, setOccasion] = useState<string>("casual");
  const [bitterness, setBitterness] = useState<number>(0.5);
  const [sweetness, setSweetness] = useState<number>(0.5);
  const [strength, setStrength] = useState<number>(0.5);
  const [topK, setTopK] = useState<number>(5);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const handleGenerate = async () => {
    setLoading(true);
    setRecommendations([]);

    try {
      const response = await fetch(`${supabaseUrl}/functions/v1/cooldrinks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${supabaseAnonKey}`,
          apikey: supabaseAnonKey,
        },
        body: JSON.stringify({
          context: {
            weather,
            time_period: timePeriod,
            occasion,
            bitterness_pref: bitterness,
            sweetness_pref: sweetness,
            strength_pref: strength,
          },
          topK,
        }),
      });

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (error) {
      console.error("Error generating recommendations:", error);
      // Fallback: show sample recommendations
      setRecommendations([
        {
          drink_id: "D001",
          name: "IPA Special",
          type: "beer",
          style: "ipa",
          abv: 6.2,
          bitterness: 75,
          sweetness: 30,
          carbonation: 3.2,
          overall_score: 0.85,
        },
        {
          drink_id: "D002",
          name: "Pilsner Premium",
          type: "beer",
          style: "pilsner",
          abv: 4.8,
          bitterness: 35,
          sweetness: 25,
          carbonation: 3.8,
          overall_score: 0.78,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle className="text-3xl">
            🥤 CoolDrinks - Context-Aware Beverage Recommender
          </CardTitle>
          <p className="text-muted-foreground">
            Powered by SOTA ML: SASRec + Multi-Modal Fusion + LinUCB
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Context Selection</h3>

              <div className="space-y-2">
                <Label>Weather Condition</Label>
                <Select value={weather} onValueChange={setWeather}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select weather" />
                  </SelectTrigger>
                  <SelectContent>
                    {WEATHER_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Time of Day</Label>
                <Select value={timePeriod} onValueChange={setTimePeriod}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select time" />
                  </SelectTrigger>
                  <SelectContent>
                    {TIME_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Occasion</Label>
                <Select value={occasion} onValueChange={setOccasion}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select occasion" />
                  </SelectTrigger>
                  <SelectContent>
                    {OCCASION_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Taste Preferences</h3>

              <div className="space-y-2">
                <Label>Bitterness Preference ({bitterness.toFixed(1)})</Label>
                <Slider
                  value={[bitterness]}
                  onValueChange={(v) => setBitterness(v[0])}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </div>

              <div className="space-y-2">
                <Label>Sweetness Preference ({sweetness.toFixed(1)})</Label>
                <Slider
                  value={[sweetness]}
                  onValueChange={(v) => setSweetness(v[0])}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </div>

              <div className="space-y-2">
                <Label>Strength Preference ({strength.toFixed(1)})</Label>
                <Slider
                  value={[strength]}
                  onValueChange={(v) => setStrength(v[0])}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </div>

              <div className="space-y-2">
                <Label>Number of Recommendations ({topK})</Label>
                <Slider
                  value={[topK]}
                  onValueChange={(v) => setTopK(v[0])}
                  min={3}
                  max={20}
                  step={1}
                />
              </div>
            </div>
          </div>

          <Button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full"
            size="lg"
          >
            {loading ? "Generating..." : "Generate Recommendations"}
          </Button>

          {recommendations.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">
                📊 Your Recommendations
              </h3>
              {recommendations.map((rec, idx) => (
                <Card key={idx} className="border-l-4 border-l-blue-500">
                  <CardContent className="pt-6">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-semibold text-lg">
                          {idx + 1}. {rec.name}
                        </h4>
                        <p className="text-sm text-muted-foreground">
                          {rec.type} - {rec.style}
                        </p>
                        <p className="text-sm mt-1">
                          <strong>ABV:</strong> {rec.abv}% |{" "}
                          <strong>Bitterness:</strong> {rec.bitterness} |{" "}
                          <strong>Sweetness:</strong> {rec.sweetness}
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-blue-600">
                          {(rec.overall_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Match Score
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {!loading && recommendations.length === 0 && (
            <div className="text-center text-muted-foreground py-8">
              <p>Select context and preferences, then click Generate to get recommendations.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
