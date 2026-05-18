import React from 'react';

const KPI = ({ title, value, icon, trend }) => {
  const titleKey = String(title || '').toLowerCase();
  const numericValue = parseFloat(String(value).replace(/[^0-9.]/g, ''));

  let semantic = 'medium';
  if (titleKey.includes('churn') || titleKey.includes('risk')) {
    semantic = Number.isFinite(numericValue) && numericValue >= 30 ? 'high' : Number.isFinite(numericValue) && numericValue <= 15 ? 'low' : 'medium';
  } else if (titleKey.includes('retain') || titleKey.includes('customer')) {
    semantic = 'low';
  } else if (titleKey.includes('revenue')) {
    semantic = Number.isFinite(numericValue) && numericValue >= 1000 ? 'low' : 'medium';
  }

  const palette = {
    low: { color: '#12B2C1', label: 'LOW', arrow: '↓' },
    medium: { color: '#FFB162', label: 'MED', arrow: '→' },
    high: { color: '#FF4D4D', label: 'HIGH', arrow: '↑' },
  }[semantic];

  return (
  <div className="glow-card backdrop-blur-sm p-6 rounded-2xl transition-all group" style={{ borderLeft: `2px solid ${palette.color}` }}>
    <div className="flex justify-between items-start mb-4">
      <div className="p-3 rounded-xl group-hover:scale-110 transition-transform" style={{ backgroundColor: `${palette.color}1f`, color: palette.color }}>
        {icon}
      </div>
      <span className="text-[10px] font-black px-2 py-1 rounded-full uppercase tracking-[0.12em]" style={{ backgroundColor: `${palette.color}22`, color: palette.color }}>
        {palette.arrow} {palette.label}
      </span>
    </div>
    <h3 className="text-[#A8CFCF] text-sm font-medium uppercase tracking-wider mb-1">{title}</h3>
    <div className="text-3xl font-bold text-[#E5F9F8] tracking-tight">{value}</div>
  </div>
  );
};

export default KPI;
