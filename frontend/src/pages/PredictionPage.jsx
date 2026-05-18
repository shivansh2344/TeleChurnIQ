import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Brain, ChevronRight, Activity, TrendingUp, User, Shield, HelpCircle } from 'lucide-react'
import { DonutChart } from '../components/Charts'
import LiquidEther from '../components/LiquidEther'
import Footer from "../components/Footer";

const FORM_SECTIONS = [
  {
    title: 'Customer Basics',
    icon: <User size={20} />,
    fields: [
      { name: 'Age', label: 'Age', type: 'range', min: 18, max: 100, col: 'span-2' },
      { name: 'Gender', label: 'Gender', type: 'select', options: [{label: 'Male', value: 1}, {label: 'Female', value: 0}], col: 'span-1' },
      { name: 'TenureinMonths', label: 'Tenure (Months)', type: 'number', placeholder: 'e.g. 24', col: 'span-1' },
    ]
  },
  {
    title: 'Subscription & Billing',
    icon: <TrendingUp size={20} />,
    fields: [
      { name: 'Contract', label: 'Contract Type', type: 'select', options: [{label: 'Month-to-Month', value: 0}, {label: 'One Year', value: 1}, {label: 'Two Year', value: 2}], col: 'span-1' },
      { name: 'MonthlyCharge', label: 'Monthly Charge (₹)', type: 'number', placeholder: 'e.g. 85.0', col: 'span-1' },
      { name: 'InternetService', label: 'Internet Service', type: 'select', options: [{label: 'Yes', value: 1}, {label: 'No', value: 0}], col: 'span-1' },
      { name: 'PaymentMethod', label: 'Payment Method', type: 'select', options: [{label: 'Bank Transfer', value: 0}, {label: 'Credit Card', value: 1}, {label: 'Mailed Check', value: 2}], col: 'span-1' },
    ]
  },
  {
    title: 'Behavior & Risk Signals',
    icon: <Activity size={20} />,
    fields: [
      { name: 'EngagementScore', label: 'Engagement Score', type: 'range', min: 1, max: 5, col: 'span-2' },
      { name: 'OnlineSecurity', label: 'Online Security', type: 'toggle', col: 'span-1' },
      { name: 'PremiumTechSupport', label: 'Tech Support', type: 'toggle', col: 'span-1' },
    ]
  }
]

const INITIAL_STATE = {
  Age: 40,
  Gender: 1,
  TenureinMonths: 12,
  Contract: 0,
  MonthlyCharge: 70.0,
  InternetService: 1,
  PaymentMethod: 1,
  EngagementScore: 3,
  OnlineSecurity: 0,
  PremiumTechSupport: 0
}

const PredictionPage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState(INITIAL_STATE)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }))
  }

  const handleToggle = (name) => {
    setFormData(prev => ({
      ...prev,
      [name]: prev[name] === 1 ? 0 : 1
    }))
  }

  const handlePredict = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    // Keep the UI minimal: backend expands this into the full model feature set.
    const payload = {
      ...formData,
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) throw new Error('Prediction request failed')
      const data = await response.json()
      setResult(data)
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const topFeatures = result?.top_features?.length
    ? result.top_features
    : result?.top_contributors?.length
      ? result.top_contributors
      : result?.explanation?.top_contributors?.length
        ? result.explanation.top_contributors
        : null;

  const reasonData = topFeatures?.length
    ? topFeatures
        .slice(0, 4)
        .map((item) => ({
          name: item.display_name || item.feature,
          value: Math.abs(item.shap_value),
        }))
    : [];

  const primaryRiskDrivers = topFeatures?.filter((item) => item.shap_value > 0).slice(0, 4) || []
  const protectiveFactors = topFeatures?.filter((item) => item.shap_value < 0).slice(0, 4) || []
  const modelInterpretation = result?.explanation
    ? result.explanation
        .split('. ')
        .map((line) => line.trim().replace(/\.$/, ''))
        .filter(Boolean)
        .slice(0, 3)
    : []

  return (
    <div className="min-h-screen bg-[#05080F] text-[#E5F9F8] selection:bg-[#12B2C1]/30">
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0">
          <LiquidEther
            colors={["#12B2C1", "#0D8A9E", "#05080F"]}
            resolution={0.4}
            autoIntensity={0.55}
            autoSpeed={0.15}
            mouseForce={7}
            cursorSize={150}
            takeoverDuration={0.2}
          />
        </div>
        <div className="absolute inset-0 bg-[#05080F]/58" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-12">
        {/* Navigation */}
        <div className="flex justify-between items-center mb-12">
          <button 
            onClick={() => navigate('/')}
            className="group flex items-center gap-2 text-[#A8CFCF] hover:text-[#E5F9F8] transition-colors font-bold text-sm tracking-widest uppercase"
          >
            <div className="p-2 bg-[#1F2B2D] rounded-lg group-hover:bg-[#273638] transition-colors border border-[#23717B]/35">
              <ArrowLeft size={16} />
            </div>
            Back to Intelligence
          </button>
          <div className="glow-heading text-xl font-black text-[#E5F9F8] tracking-tighter">TeleChurnIQ</div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start">
          <section className="xl:col-span-5 glow-card rounded-3xl p-8">
            <div className="mb-8">
              <h1 className="glow-heading text-4xl font-black text-[#E5F9F8] tracking-tight">Predictive Analysis Console</h1>
              <p className="text-[#A8CFCF] mt-3 text-base leading-relaxed">
                Submit customer attributes to run risk inference and explain model behavior with transparent drivers.
              </p>
            </div>

            <form onSubmit={handlePredict} className="space-y-7">
              {FORM_SECTIONS.map((section, idx) => (
                <div key={idx} className="rounded-2xl border border-[#23717B]/45 bg-[#1F2B2D]/65 p-6">
                  <h2 className="text-sm font-black text-[#E5F9F8] mb-6 flex items-center gap-3 uppercase tracking-[0.16em]">
                    <div className="p-2 rounded-lg bg-[#0D8A9E]/18 text-[#12B2C1]">
                      {section.icon}
                    </div>
                    {section.title}
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {section.fields.map(field => (
                      <div key={field.name} className={field.col}>
                        <div className="flex justify-between items-center mb-2.5">
                          <label className="text-[11px] font-black text-[#A8CFCF] uppercase tracking-[0.12em]">
                            {field.label}
                          </label>
                          {field.type === 'range' && (
                            <span className="text-[#12B2C1] font-black text-xs bg-[#0D8A9E]/18 px-2 py-1 rounded-md">
                              {formData[field.name]}
                            </span>
                          )}
                        </div>

                        {field.type === 'select' ? (
                          <div className="relative group">
                            <select
                              name={field.name}
                              value={formData[field.name]}
                              onChange={handleChange}
                              className="w-full bg-[#05080F]/76 border border-[#23717B]/55 text-[#E5F9F8] rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-[#12B2C1]/45 focus:border-[#12B2C1] transition-all appearance-none font-semibold text-sm"
                            >
                              {field.options.map(opt => (
                                <option key={opt.value} value={opt.value} className="bg-[#05080F]">{opt.label}</option>
                              ))}
                            </select>
                            <div className="absolute right-3.5 top-1/2 -translate-y-1/2 pointer-events-none text-[#A8CFCF]/80 group-hover:text-[#E5F9F8] transition-colors">
                              <ChevronRight size={15} className="rotate-90" />
                            </div>
                          </div>
                        ) : field.type === 'range' ? (
                          <input
                            type="range"
                            name={field.name}
                            min={field.min}
                            max={field.max}
                            value={formData[field.name]}
                            onChange={handleChange}
                            className="w-full h-1.5 bg-[#05080F] rounded-full appearance-none cursor-pointer accent-[#12B2C1] hover:accent-[#12B2C1] transition-all"
                          />
                        ) : field.type === 'toggle' ? (
                          <button
                            type="button"
                            onClick={() => handleToggle(field.name)}
                            className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors focus:outline-none ${formData[field.name] === 1 ? 'bg-[#0D8A9E]' : 'bg-[#05080F]'}`}
                          >
                            <span
                              className={`inline-block h-5 w-5 transform rounded-full bg-[#E5F9F8] transition-transform ${formData[field.name] === 1 ? 'translate-x-6' : 'translate-x-1'}`}
                            />
                          </button>
                        ) : (
                          <input
                            type={field.type}
                            name={field.name}
                            value={formData[field.name]}
                            onChange={handleChange}
                            placeholder={field.placeholder}
                            className="w-full bg-[#05080F]/76 border border-[#23717B]/55 text-[#E5F9F8] rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-[#12B2C1]/45 focus:border-[#12B2C1] transition-all font-semibold text-sm"
                            required
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              <button
                type="submit"
                disabled={loading}
                className="glow-button w-full py-4 font-black rounded-2xl transition duration-300 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 text-lg tracking-tight"
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-4 border-[#05080F]/25 border-t-[#05080F] rounded-full animate-spin"></div>
                    Running Analysis...
                  </>
                ) : (
                  <>
                    <Brain size={22} />
                    Predict Customer Risk
                  </>
                )}
              </button>
            </form>
          </section>

          <section className="xl:col-span-7 space-y-7">
            {error ? (
              <div className="glow-card rounded-3xl p-7 border-[#23717B]/70 bg-[#1F2B2D]/78">
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-[#23717B]/30 rounded-lg text-[#E5F9F8]">
                    <Shield size={18} />
                  </div>
                  <div>
                    <h3 className="text-xs uppercase tracking-[0.2em] font-black text-[#A8CFCF] mb-2">System Error</h3>
                    <p className="text-[#E5F9F8] font-semibold">{error}</p>
                  </div>
                </div>
              </div>
            ) : null}

            <div className={`glow-card rounded-3xl p-8 ${result ? '' : 'min-h-[320px] flex items-center justify-center text-center'}`}>
              {!result && !loading ? (
                <div>
                  <div className="w-20 h-20 bg-[#0D8A9E]/16 rounded-full flex items-center justify-center mx-auto mb-5 text-[#12B2C1] border border-[#23717B]/55">
                    <HelpCircle size={42} strokeWidth={1.5} />
                  </div>
                  <h3 className="glow-heading text-2xl font-black text-[#E5F9F8] mb-2">AI Engine Ready</h3>
                  <p className="text-[#A8CFCF]">Provide customer inputs to generate risk score and model interpretation.</p>
                </div>
              ) : null}

              {result ? (
                <div className="text-center">
                  <h2 className={`glow-heading text-6xl font-black tracking-tight ${result.prediction === 1 ? 'text-[#23717B]' : 'text-[#12B2C1]'}`}>
                    {result.prediction === 1 ? 'HIGH RISK' : 'STABLE'}
                  </h2>
                  <div className="mt-3 text-[#A8CFCF] text-lg font-semibold uppercase tracking-[0.14em]">Predicted Customer State</div>

                  <div className="mt-8">
                    <p className="text-sm text-[#A8CFCF] uppercase tracking-[0.15em] mb-3">Probability</p>
                    <p className="text-5xl font-black text-[#E5F9F8]">{(result.probability * 100).toFixed(1)}%</p>
                  </div>

                  <div className="mt-8 h-3 w-full bg-[#05080F]/80 rounded-full overflow-hidden border border-[#23717B]/45">
                    <div
                      className={`h-full transition-all duration-[1200ms] ease-out ${result.prediction === 1 ? 'bg-gradient-to-r from-[#23717B] to-[#0D8A9E]' : 'bg-gradient-to-r from-[#0D8A9E] to-[#12B2C1]'}`}
                      style={{ width: `${result.probability * 100}%`, boxShadow: '0 0 16px rgba(18,178,193,0.6)' }}
                    />
                  </div>
                </div>
              ) : null}
            </div>

            {result ? (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                <div className="glow-card rounded-2xl p-5">
                  <h3 className="text-xs font-black uppercase tracking-[0.18em] text-[#12B2C1] mb-4">Primary Risk Drivers</h3>
                  {primaryRiskDrivers.length ? (
                    <ul className="space-y-3">
                      {primaryRiskDrivers.map((item, idx) => (
                        <li key={`risk-${item.feature}-${idx}`} className="rounded-xl border border-[#23717B]/40 bg-[#05080F]/45 p-3">
                          <div className="text-[#E5F9F8] text-sm font-bold">{item.display_name || item.feature}</div>
                          <div className="text-[#A8CFCF] text-xs mt-1">{item.value_text || item.value}</div>
                          <div className="text-[#23717B] text-xs font-black mt-2 uppercase tracking-[0.12em]">↑ increases risk</div>
                        </li>
                      ))}
                    </ul>
                  ) : <p className="text-[#A8CFCF] text-sm">No strong upward risk drivers detected.</p>}
                </div>

                <div className="glow-card rounded-2xl p-5">
                  <h3 className="text-xs font-black uppercase tracking-[0.18em] text-[#12B2C1] mb-4">Protective Factors</h3>
                  {protectiveFactors.length ? (
                    <ul className="space-y-3">
                      {protectiveFactors.map((item, idx) => (
                        <li key={`safe-${item.feature}-${idx}`} className="rounded-xl border border-[#23717B]/40 bg-[#05080F]/45 p-3">
                          <div className="text-[#E5F9F8] text-sm font-bold">{item.display_name || item.feature}</div>
                          <div className="text-[#A8CFCF] text-xs mt-1">{item.value_text || item.value}</div>
                          <div className="text-[#12B2C1] text-xs font-black mt-2 uppercase tracking-[0.12em]">↓ reduces risk</div>
                        </li>
                      ))}
                    </ul>
                  ) : <p className="text-[#A8CFCF] text-sm">No major protective factors surfaced.</p>}
                </div>

                <div className="glow-card rounded-2xl p-5">
                  <h3 className="text-xs font-black uppercase tracking-[0.18em] text-[#12B2C1] mb-4">Model Interpretation</h3>
                  {modelInterpretation.length ? (
                    <ul className="space-y-3">
                      {modelInterpretation.map((line, idx) => (
                        <li key={`interp-${idx}`} className="rounded-xl border border-[#23717B]/38 bg-[#05080F]/45 p-3 text-sm text-[#A8CFCF] leading-relaxed">
                          {line}
                        </li>
                      ))}
                    </ul>
                  ) : <p className="text-[#A8CFCF] text-sm">Interpretation will appear after prediction.</p>}
                </div>
              </div>
            ) : null}

            {reasonData.length ? <DonutChart data={reasonData} title="Risk Factor Contribution" /> : null}
          </section>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default PredictionPage;
