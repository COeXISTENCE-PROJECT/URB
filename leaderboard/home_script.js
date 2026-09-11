    const homeResultData = {
      st_arnoult: {
        title: 'St. Arnoult (Small)',
        description: 'Mean CAV travel times. Lower is better. In small networks, QMIX occasionally beats humans.',
        winrate: '80%',
        values: { human: 3.15, aon: 3.01, random: 3.58, qmix: 3.21, ippo: 3.33, iql: 3.53, mappo: 3.51 },
      },
      provins: {
        title: 'Provins (Medium)',
        description: 'Mean CAV travel times. Lower is better. In larger networks, MARL algorithms struggle to match human efficiency.',
        winrate: '0%',
        values: { human: 2.80, aon: 2.76, random: 3.04, qmix: 3.14, ippo: 2.98, iql: 3.01, mappo: 3.05 },
      },
      ingolstadt: {
        title: 'Ingolstadt (Large)',
        description: 'Mean CAV travel times. Lower is better. In larger networks, MARL algorithms struggle to match human efficiency.',
        winrate: '0%',
        values: { human: 4.21, aon: 4.37, random: 4.81, qmix: 4.87, ippo: 4.71, iql: 4.81, mappo: 4.82 },
      },
    };

    const homeScenarioButtons = Array.from(document.querySelectorAll('[data-home-scenario]'));
    const homeResultRows = Array.from(document.querySelectorAll('[data-home-series]'));
    const homeResultTitle = document.getElementById('home-result-title');
    const homeResultDescription = document.getElementById('home-result-description');
    const homeResultWinrate = document.getElementById('home-result-winrate');

    function renderHomeResult(scenario) {
      const result = homeResultData[scenario];
      if (!result) return;

      const maxValue = Math.max(...Object.values(result.values)) * 1.05;
      homeResultTitle.textContent = result.title;
      homeResultDescription.textContent = result.description;
      homeResultWinrate.textContent = result.winrate;

      homeResultRows.forEach((row) => {
        const key = row.dataset.homeSeries;
        const value = result.values[key];
        const valueLabel = row.querySelector('strong');
        const bar = row.querySelector('.home-result-track i');
        valueLabel.textContent = `${value.toFixed(2)}m`;
        bar.style.width = `${(value / maxValue) * 100}%`;
      });

      homeScenarioButtons.forEach((button) => {
        const selected = button.dataset.homeScenario === scenario;
        button.classList.toggle('is-active', selected);
        button.setAttribute('aria-pressed', String(selected));
      });
    }

    homeScenarioButtons.forEach((button) => {
      button.addEventListener('click', () => renderHomeResult(button.dataset.homeScenario));
    });
    renderHomeResult('st_arnoult');

    const homeCopyCitation = document.getElementById('home-copy-citation');
    const homeCopyCitationIcon = document.getElementById('home-copy-citation-icon');
    const homeCitationText = document.getElementById('home-citation-text');

    async function copyHomeCitation() {
      const citation = homeCitationText.textContent;
      try {
        await navigator.clipboard.writeText(citation);
      } catch (error) {
        const textarea = document.createElement('textarea');
        textarea.value = citation;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        textarea.remove();
      }

      homeCopyCitationIcon.setAttribute('href', '#icon-check');
      homeCopyCitation.setAttribute('aria-label', 'Citation copied');
      homeCopyCitation.title = 'Citation copied';
      window.setTimeout(() => {
        homeCopyCitationIcon.setAttribute('href', '#icon-copy');
        homeCopyCitation.setAttribute('aria-label', 'Copy BibTeX citation');
        homeCopyCitation.title = 'Copy BibTeX citation';
      }, 1800);
    }

    homeCopyCitation.addEventListener('click', copyHomeCitation);
