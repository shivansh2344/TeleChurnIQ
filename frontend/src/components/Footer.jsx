export default function Footer() {
  return (
    <footer className="relative border-t border-[#23717B]/35 mt-20">
      
      {/* subtle gradient glow */}
      <div className="absolute inset-0 bg-gradient-to-r from-[#12B2C1]/10 via-[#0D8A9E]/10 to-transparent pointer-events-none" />

      <div className="relative flex flex-col md:flex-row justify-between items-center px-6 py-6 text-sm text-[#A8CFCF]">
        
        {/* LEFT */}
        <div className="mb-2 md:mb-0">
          © 2026 <span className="text-[#E5F9F8] font-semibold">TeleChurnIQ</span>
        </div>

        {/* CENTER */}
        <div className="text-center text-xs text-[#A8CFCF]/80">
          AI-powered customer intelligence
        </div>

        {/* RIGHT */}
        <div className="flex items-center gap-3">
          <span>
            Built by{" "}
            <span className="text-[#12B2C1] font-medium hover:text-[#7bdddf] transition duration-300">
              Shivansh Garg
            </span>
          </span>

          {/* subtle dot */}
          <span className="w-1 h-1 bg-[#A8CFCF]/55 rounded-full"></span>

          {/* optional links */}
          <a
            href="https://github.com/shivansh2344"
            className="hover:text-[#E5F9F8] transition"
          >
            GitHub
          </a>

          <a
            href="https://www.linkedin.com/in/shivansh2344/"
            className="hover:text-[#E5F9F8] transition"
          >
            LinkedIn
          </a>
        </div>
      </div>
    </footer>
  );
}