import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Papa from 'papaparse';
import { Users, UserX, IndianRupee, Loader2 } from 'lucide-react';

import LiquidEther from '../components/LiquidEther';
import KPI from '../components/KPI';
import { DonutChart, ComparisonBarChart } from '../components/Charts';
import Footer from "../components/Footer";
import LineWaves from "../components/LineWaves";

const LandingPage = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  const [stats, setStats] = useState({
    totalCustomers: 0,
    churnRate: '0%',
    avgRevenue: '₹0',
    churnData: [],
    contractData: []
  });

  useEffect(() => {
    Papa.parse('/data/telco.csv', {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const raw = results.data;

        const data = raw.map(row => ({
          ...row,
          ChurnValue: Number(row.Churn || 0),
          Revenue: Number(row.AvgRevenue || 0),
          Contract: String(row.Contract || '')
        }));

        const total = data.length;
        const churned = data.reduce((a, b) => a + b.ChurnValue, 0);
        const revenue = data.reduce((a, b) => a + b.Revenue, 0);

        const churnRate = total > 0 ? ((churned / total) * 100).toFixed(1) : 0;
        const avgRevenue = total > 0 ? (revenue / total).toFixed(2) : 0;

        const churnPie = [
          { name: 'Retained', value: total - churned },
          { name: 'Churned', value: churned }
        ];

        const contractLabels = {
          '0': 'Month-to-Month',
          '1': 'One Year',
          '2': 'Two Year'
        };

        const contractBar = ['0', '1', '2'].map(type => {
          const group = data.filter(d => String(d.Contract) === type);
          const churn = group.reduce((a, b) => a + b.ChurnValue, 0);
          return {
            name: contractLabels[type],
            churnRate: group.length ? parseFloat(((churn / group.length) * 100).toFixed(1)) : 0
          };
        });

        setStats({
          totalCustomers: total.toLocaleString(),
          churnRate: `${churnRate}%`,
          avgRevenue: `₹${avgRevenue}`,
          churnData: churnPie,
          contractData: contractBar
        });

        setLoading(false);
      }
    });
  }, []);

  return (
    <div className="bg-[#05080F] text-[#E5F9F8] min-h-screen">

      {/* FIXED AMBIENT BACKGROUND */}
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

      {/* HERO */}
      <section className="relative min-h-[92vh] flex items-center justify-center overflow-hidden isolate z-30 bg-[#05080F]">

        {/* BACKGROUND */}
        <div className="absolute inset-0 z-0 pointer-events-none">
          <div className="absolute inset-0 opacity-[0.64]">
            <LineWaves
              brightness={0.145}
              color1="#12B2C1"
              color2="#0D8A9E"
              color3="#A8CFCF"
              enableMouseInteraction={false}
              speed={0.16}
              warpIntensity={0.72}
            />
          </div>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_32%,rgba(18,178,193,0.22)_0%,rgba(5,8,15,0.88)_52%,rgba(5,8,15,1)_100%)]"></div>
          <div className="absolute inset-0 bg-[#05080F]/52"></div>
          <div className="absolute inset-x-0 bottom-0 h-44 bg-gradient-to-b from-transparent via-[#05080F]/84 to-[#05080F]"></div>
        </div>

        {/* CONTENT */}
        <div className="relative z-10 text-center px-6 max-w-5xl pt-24 pb-24">

          <p className="text-[11px] md:text-xs uppercase tracking-[0.24em] text-[#A8CFCF] font-black mb-6">
            AI-Powered Churn Intelligence
          </p>

          <h1 className="glow-heading text-4xl sm:text-5xl md:text-7xl font-black leading-[1.04] tracking-tight text-[#E5F9F8]">
            TeleChurnIQ
            <br /> Predict churn before it happens.
          </h1>

          <p className="mt-8 text-base md:text-xl text-[#A8CFCF] max-w-3xl mx-auto leading-relaxed">
            Turn churn signals into retention strategy with AI-driven risk scoring, decision-ready explanations, and proactive customer intervention.
          </p>

          <div className="mt-12 flex flex-col sm:flex-row justify-center gap-4">
            <button
              onClick={() => navigate('/predict')}
              className="glow-button px-8 py-4 font-black rounded-xl transition text-[#05080F] hover:scale-[1.02]"
            >
              Predict Customer Risk
            </button>

            <a
              href="#intelligence"
              className="px-8 py-4 border border-[#23717B]/50 hover:border-[#12B2C1] text-[#E5F9F8] rounded-xl transition"
            >
              Explore Intelligence
            </a>
          </div>

          <p className="mt-7 text-sm text-[#A8CFCF]/85 tracking-wide">
            One prediction can save a quarter of revenue when you act before cancellation intent peaks.
          </p>

        </div>
      </section>

      <div className="relative z-10">

      {/* WHY TELECHURNIQ */}
      <section className="max-w-6xl mx-auto px-6 py-20 text-center">

        <h2 className="glow-heading text-4xl font-bold mb-6 text-[#E5F9F8]">
          Why TeleChurnIQ?
        </h2>

        <p className="text-[#A8CFCF] max-w-3xl mx-auto mb-12">
          Most companies react after customers leave. TeleChurnIQ flips the game —
          it predicts risk early and tells you exactly what to fix.
        </p>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="glow-card p-6 rounded-xl">
            <h3 className="text-[#12B2C1] font-semibold mb-2">Predictive Intelligence</h3>
            <p className="text-[#A8CFCF] text-sm">
              ML-powered churn prediction based on behavioral patterns.
            </p>
          </div>

          <div className="glow-card p-6 rounded-xl">
            <h3 className="text-[#12B2C1] font-semibold mb-2">Actionable Insights</h3>
            <p className="text-[#A8CFCF] text-sm">
              Not just predictions — we tell you what actions to take.
            </p>
          </div>

          <div className="glow-card p-6 rounded-xl">
            <h3 className="text-[#12B2C1] font-semibold mb-2">Real Business Impact</h3>
            <p className="text-[#A8CFCF] text-sm">
              Reduce churn, increase retention, and boost revenue.
            </p>
          </div>
        </div>
      </section>

      {/* DASHBOARD */}
      <section id="intelligence" className="max-w-7xl mx-auto px-6 py-20">

        <h2 className="glow-heading text-4xl font-bold mb-10 text-[#E5F9F8]">
          Real-time Intelligence
        </h2>

        {loading ? (
          <div className="flex justify-center py-20">
            <Loader2 className="animate-spin text-[#12B2C1]" size={40} />
          </div>
        ) : (
          <>
            <div className="grid md:grid-cols-3 gap-6 mb-10">
              <KPI title="Customers" value={stats.totalCustomers} icon={<Users />} />
              <KPI title="Churn Rate" value={stats.churnRate} icon={<UserX />} />
              <KPI title="Revenue" value={stats.avgRevenue} icon={<IndianRupee />} />
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <DonutChart data={stats.churnData} title="Churn Distribution" />
              <ComparisonBarChart data={stats.contractData} title="Contract Risk" xKey="name" yKey="churnRate" />
            </div>
          </>
        )}
      </section>

      <Footer />
      </div>
    </div>
  );
};

export default LandingPage;